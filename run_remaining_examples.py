#!/usr/bin/env python3
import json, subprocess, time
from pathlib import Path
ROOT=Path('/home/francois/projects/minimax-h3-runner'); BASE='http://127.0.0.1:8988'
jobs=[('image-plus-prompt-5s-10fps',{'prompt':'The fox slowly turns its head while leaves move in the breeze','first_frame':str(ROOT/'input/fox-first.png'),'duration':5,'fps':10,'width':384,'height':224}),('long-30s-10fps',{'prompt':'A dancer performs a graceful energetic routine in a studio','first_frame':str(ROOT/'input/dancer-source.jpg'),'duration':30,'fps':10,'width':256,'height':384,'audio':'first'})]
def get(args):
 p=subprocess.run(['curl','-fsS','--max-time','20',*args],capture_output=True,text=True,check=True); return json.loads(p.stdout)
results=[]
for name,payload in jobs:
 print('SUBMIT',name,flush=True); r=get(['-X','POST',BASE+'/generate','-H','Content-Type: application/json','--data',json.dumps(payload)]); j=r['job']; print('JOB',name,j['id'],j['output'],flush=True)
 while True:
  j=get([f"{BASE}/status?job={j['id']}"])['job']; print('STATUS',name,j['id'],j['state'],flush=True)
  if j['state'] in ('done','failed'):
   results.append({'name':name,**j}); break
  time.sleep(10)
 if j['state']=='failed': raise SystemExit(j.get('error','job failed'))
Path(ROOT/'server-example-results-remaining.json').write_text(json.dumps(results,indent=2)+'\n')
print('DONE',flush=True)
