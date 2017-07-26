from datetime import datetime
import json
import resume_parser
import urllib
import os

status = "processing"
def MyProcFileFunction(docId, tenantId, filePath):
    global status
    status="File Downloading"
    ext = filePath.split('.')[-1]
    
    directory = "data/" + tenantId + "/"
    resumePath = directory + docId + "." + ext
    if not os.path.exists(directory):
         os.makedirs(directory)
    urllib.urlretrieve(filePath, resumePath)
    status="Parsing Resume"
    
    resume_parser.upsert_resume(resumePath)
    status = "Deleting Resume"
    os.remove(resumePath)
    status = "Resume Deleted"
    
    return status

with open('list2.json') as data_file:
    global status
    log = open("./downloadAndProcessFiles.py.log", "a+")
    log.write(str(datetime.now()) + ": -------------------- Service Invoked-----------\n")
    data = json.load(data_file)
    
    for record in data["data"]:
        try:
                status = "Processing"
		status = MyProcFileFunction(record["doc_id"] , record["tenant_id"], record["file_path"])
                status = "Processed"
                log.write(str(datetime.now()) + ": " + status + " file " + record["file_path"] + "\n")
	except Exception as e:
		log.write("Error occured -- doc_id:" + record["doc_id"] + ", tenant_id:" + record["tenant_id"]  + ", file_path:" +  record["file_path"] + ", Status:" + status + " : " + str(e) + "\n")
    log.close()
