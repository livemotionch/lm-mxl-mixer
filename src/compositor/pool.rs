//! A small pool for drawing strips in parallel.
//!
//! Idle workers sleep on a condition variable. (rayon's workers spin for a
//! while before sleeping; at 50 frames per second that spinning cost almost
//! half as much CPU again as the drawing.) The calling thread draws strips
//! too, and returns once every worker is done with the frame.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

type Job = dyn Fn(usize) + Sync;

/// Lifetime-erased pointer to the current job; only dereferenced while
/// [`StripPool::run`] waits for the workers.
#[derive(Clone, Copy)]
struct JobPtr(*const Job);

// SAFETY: the job is `Sync`, and `run` keeps it alive until no worker uses it.
unsafe impl Send for JobPtr {}

struct State {
    generation: u64,
    job: Option<JobPtr>,
    tasks: usize,
    /// Workers not yet done with the current generation
    active: usize,
    /// A worker's task panicked in the current generation
    panicked: bool,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    work: Condvar,
    done: Condvar,
    next: AtomicUsize,
}

pub struct StripPool {
    shared: Arc<Shared>,
    workers: usize,
    /// One frame at a time
    running: Mutex<()>,
}

impl StripPool {
    /// A pool with `threads` threads in total, the caller included.
    pub fn new(threads: usize) -> StripPool {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                generation: 0,
                job: None,
                tasks: 0,
                active: 0,
                panicked: false,
                shutdown: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
            next: AtomicUsize::new(0),
        });
        let workers = threads.max(1) - 1;
        for i in 0..workers {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name(format!("compositor-{i}"))
                .spawn(move || worker(&shared))
                .expect("mixer thread");
        }
        StripPool {
            shared,
            workers,
            running: Mutex::new(()),
        }
    }

    /// Threads drawing, the caller included.
    pub fn threads(&self) -> usize {
        self.workers + 1
    }

    /// Run `f(0)`, ..., `f(tasks - 1)` on the pool and the calling thread.
    /// Returns (or unwinds, if a task panicked) only once no worker uses `f`
    /// any more.
    pub fn run<'a>(&self, tasks: usize, f: &'a (dyn Fn(usize) + Sync + 'a)) {
        if self.workers == 0 || tasks <= 1 {
            (0..tasks).for_each(f);
            return;
        }
        // Another composition is drawing on the pool: draw on this thread.
        let _one = match self.running.try_lock() {
            Ok(guard) => guard,
            // (Poisoned by an earlier panic: nothing to recover.)
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                (0..tasks).for_each(f);
                return;
            }
        };
        // SAFETY: erases the lifetime; the pointer is only used until all
        // workers report done below, before `f` goes out of scope.
        let ptr = JobPtr(unsafe {
            std::mem::transmute::<&'a (dyn Fn(usize) + Sync + 'a), &'static Job>(f)
        });
        {
            let mut st = lock(&self.shared.state);
            st.generation += 1;
            st.job = Some(ptr);
            st.tasks = tasks;
            st.active = self.workers;
            st.panicked = false;
            self.shared.next.store(0, Ordering::SeqCst);
            self.shared.work.notify_all();
        }
        // Waits for the workers when dropped, also while unwinding from a
        // panic in `f` on this thread: `f` (and what it borrows) must outlive
        // every use by a worker.
        let wait = WaitForWorkers(&self.shared);
        take_tasks(&self.shared, tasks, f);
        drop(wait);
        let panicked = std::mem::take(&mut lock(&self.shared.state).panicked);
        if panicked {
            panic!("compositor: a strip panicked");
        }
    }
}

/// Lock, ignoring poisoning: the state stays consistent (every update is a
/// plain assignment), and the pool must keep working after a panic.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct WaitForWorkers<'a>(&'a Shared);

impl Drop for WaitForWorkers<'_> {
    fn drop(&mut self) {
        // Workers that haven't started take no task any more.
        self.0.next.store(usize::MAX / 2, Ordering::SeqCst);
        let mut st = lock(&self.0.state);
        while st.active > 0 {
            st = self.0.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.job = None;
    }
}

fn take_tasks(shared: &Shared, tasks: usize, f: &(dyn Fn(usize) + Sync + '_)) {
    loop {
        let i = shared.next.fetch_add(1, Ordering::SeqCst);
        if i >= tasks {
            break;
        }
        f(i);
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        let (job, tasks) = {
            let mut st = lock(&shared.state);
            while st.generation == seen && !st.shutdown {
                st = shared.work.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if st.shutdown {
                return;
            }
            seen = st.generation;
            (st.job, st.tasks)
        };
        if let Some(JobPtr(job)) = job {
            // SAFETY: `run` waits for this worker before the job is dropped.
            let job = unsafe { &*job };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                take_tasks(shared, tasks, job)
            }));
            if result.is_err() {
                tracing::error!("compositor: a strip panicked");
                lock(&shared.state).panicked = true;
            }
        }
        let mut st = lock(&shared.state);
        st.active -= 1;
        if st.active == 0 {
            shared.done.notify_all();
        }
    }
}

impl Drop for StripPool {
    fn drop(&mut self) {
        lock(&self.shared.state).shutdown = true;
        self.shared.work.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_task_once() {
        let pool = StripPool::new(4);
        for tasks in [0, 1, 3, 17] {
            let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            for _ in 0..50 {
                pool.run(tasks, &|i| {
                    hits[i].fetch_add(1, Ordering::SeqCst);
                });
            }
            assert!(hits.iter().all(|h| h.load(Ordering::SeqCst) == 50));
        }
    }

    /// A panicking task doesn't let `run` return (or unwind) while workers
    /// still use the job, and the pool keeps working afterwards.
    #[test]
    fn panics_wait_for_the_workers_and_propagate() {
        let pool = StripPool::new(4);
        for panic_on_worker in [false, true] {
            let running = AtomicUsize::new(0);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.run(64, &|_| {
                    let worker = std::thread::current()
                        .name()
                        .is_some_and(|n| n.starts_with("compositor-"));
                    if worker == panic_on_worker {
                        panic!("strip");
                    }
                    running.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    running.fetch_sub(1, Ordering::SeqCst);
                });
            }));
            assert!(result.is_err(), "panic on worker: {panic_on_worker}");
            assert_eq!(
                running.load(Ordering::SeqCst),
                0,
                "a worker still runs the job"
            );
        }
        let hits = AtomicUsize::new(0);
        pool.run(32, &|_| {
            hits.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(hits.load(Ordering::SeqCst), 32);
    }
}
