#!/usr/bin/env python3
import json, subprocess, time
from pathlib import Path
root=Path('/home/francois/projects/minimax-h3-runner'); base='http://127.0.0.1:8988'
payload={'prompt':'A dancer performs a graceful energetic routine in a studio','first_frame':str(root/'input/dancer-source.jpg'),'duration':30,'fps':10,'width':256,'height':384,'audio':'first'}
def curl(args):
 p=subprocess.run(['curl','-fsS','--max-time','20',*args],capture_output=True,text=True,check=True); return json.loads(p.stdout)
r=curl(['-X','POST',base+'/generate','-H','Content-Type: application/json','--data',json.dumps(payload)])
j=r['job']; print('JOB',j['id'],j['output'],flush=True)
while True:
 j=curl([f"{base}/status?job={j['id']}"])['job']; print('STATUS',j['id'],j['state'],flush=True)
 if j['state'] in ('done','failed'):
  (root/'server-example-result-long.json').write_text(json.dumps(j,indent=2)+'\n')
  if j['state']=='failed': raise SystemExit(j.get('error','failed'))
  print('DONE',j['output'],flush=True); break
 time.sleep(10)
