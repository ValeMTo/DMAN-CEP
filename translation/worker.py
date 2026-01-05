import threading

class ThreadManager:
    def __init__(self):
        self.threads = {}
        self.results = {}

    def run_parallel(self, name, func, args):
        """
        Runs the function in a thread and stores its result.
        """
        def wrapper():
            result = func(*args)
            self.results[name] = result  # Save the return value

        thread = threading.Thread(target=wrapper)
        thread.start()
        self.threads[name] = thread

    def wait_for(self, name):
        thread = self.threads.get(name)
        if thread:
            thread.join()
        else:
            raise ValueError(f"No thread found with name: {name}")

    def get_result(self, name):
        self.wait_for(name)
        return self.results[name]
    
