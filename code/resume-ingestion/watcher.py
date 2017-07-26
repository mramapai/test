import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from resume_parser import upsert_resume, delete_resume


class MyHandler(FileSystemEventHandler):
	def on_any_event(self, event):
		pass
		# print 'on_any_event'

	def on_created(self, event):
		pass
		# print 'on_created'

	def on_deleted(self, event):
		if not event.is_directory and not os.path.basename(event.src_path).startswith('.'):
			print event.src_path + ' deleted'
			delete_resume(event.src_path)

	def on_modified(self, event):
		if not event.is_directory and not os.path.basename(event.src_path).startswith('.'):
			print event.src_path + ' modified'
			upsert_resume(event.src_path)

	def on_moved(self, event):
		# print 'on_moved'
		pass
					

if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='/mnt/s3/', recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

