import threading

_init_lock = threading.Lock()
_initialized = False


class AppStartupMiddleware:
    def __init__(self, get_response):
        global _initialized
        with _init_lock:
            if not _initialized:
                _initialized = True
                self._run_startup()
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    @staticmethod
    def _run_startup():
        import os
        from app.application.ml.load_models import load_initial_models
        from app.domain.core.config import settings

        load_initial_models()

        if os.environ.get("RUN_MAIN") == "true" or not settings.app_debug:
            from app.infrastructure.jobs.job_scheduler import job_scheduler

            job_scheduler.start()
