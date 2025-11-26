use std::{num::NonZero, sync::OnceLock, thread::available_parallelism};

pub fn tokio_runtime() -> &'static tokio::runtime::Runtime {
    static TOKIO_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    TOKIO_RUNTIME.get_or_init(|| {
        let default_parallelism = NonZero::new(2usize).unwrap();
        let parallelism = available_parallelism()
            .unwrap_or(default_parallelism)
            .max(default_parallelism);
        let tokio_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(parallelism.get())
            .enable_all()
            .build();
        tokio_rt.unwrap()
    })
}
