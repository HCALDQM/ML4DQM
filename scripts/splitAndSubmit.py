#! /usr/bin/env python

import os
import sys
import optparse
import datetime
import time

usage = "python ../scripts/splitAndSubmit.py -t hcal_dqm_offline_cfg.py -q cmscaf1nd -i fileList_JetHT_Run2017F-v1_RAW_306138.txt -p /eos/cms/store/group/dpg_hcal/comm_hcal/ML4DQM/process_2017 --ls 1,2,5-1200"

parser = optparse.OptionParser(usage)
parser.add_option("-t", "--template", dest="template",
    help="name of the template config to be used",
    )

parser.add_option('-q', '--queue',       action='store',     dest='queue',       
    help='run in batch in queue specified as option (default -q cmslong)', 
    default='cmsan',
    metavar="QUEUE")

parser.add_option("-i", "--input", dest="input",
    help="path and name of the fileList",
    )

parser.add_option("--ls", dest="lsrange",
    help="lumisection range",
    )

parser.add_option("-p", "--outpath", dest="outpath",
    help="the root file outPath",
    metavar="OUTPATH")

parser.add_option('-I', '--interactive',      
    action='store_true',
    dest='interactive',      
    help='run the jobs interactively, 2 jobs at a time',
    default=False)

(opt, args) = parser.parse_args()
################################################


###
pwd = os.environ['PWD']
current_time = datetime.datetime.now()
simpletimeMarker = "_%04d%02d%02d_%02d%02d%02d" % (current_time.year,current_time.month,current_time.day,current_time.hour,current_time.minute,current_time.second) 
timeMarker = "submit_%04d%02d%02d_%02d%02d%02d" % (current_time.year,current_time.month,current_time.day,current_time.hour,current_time.minute,current_time.second) 
workingDir = pwd+"/batch/"+timeMarker

os.system("mkdir -p "+workingDir)
os.system("mkdir -p "+opt.outpath+"/"+timeMarker)
os.system("cp "+opt.template+" "+workingDir)

template = workingDir+"/"+opt.template

xranges = [(lambda l: xrange(l[0], l[-1]+1))(map(int, r.split('-'))) for r in opt.lsrange.split(',')]
lumiList = [y for x in xranges for y in x]

run = ''
inputlist = []
secinputlist = []
njobs_list = []

inList = open(opt.input, "r")
for c,line in enumerate(inList):
    line = line.rstrip('\n')
    if c==0:
        inputlist.append(line)
        tmpLine = line.split("/")
        run = tmpLine[-4]+tmpLine[-3]
    else:
        secinputlist.append(line)


##loop over lists (one for datasets) to create splitted lists
for ls in  lumiList:
    
    os.system("mkdir "+workingDir+"/"+str(ls))
        
    with open(template) as fi:
        contents = fi.read()
        replaced_contents = contents.replace('INPUTLIST', str(inputlist))
        replaced_contents = replaced_contents.replace('SECONDARY', str(secinputlist))
        replaced_contents = replaced_contents.replace('LUMIRANGE', '"'+str(run)+':'+str(ls)+'-'+str(run)+':'+str(ls)+'"')
            
        with open(workingDir+"/"+str(ls)+"/config.py", "w") as fo:
            fo.write(replaced_contents)

        os.system("echo cd "+pwd+" > launch.sh")
        os.system("echo 'eval `scramv1 runtime -sh`\n' >> launch.sh")
        os.system("echo cd - >> launch.sh")
        os.system("echo cmsRun "+workingDir+"/"+str(ls)+"/config.py >> launch.sh")
        os.system("echo mv *"+run+"*.root "+opt.outpath+"/"+timeMarker+"/DQM_run"+run+"_ls"+str(ls)+".root >> launch.sh")
        os.system("chmod 755 launch.sh")
        os.system("mv launch.sh "+workingDir+"/"+str(ls))
        njobs_list.append("bsub -q"+opt.queue+" -o "+workingDir+"/"+str(ls)+"/log.out -e "+workingDir+"/"+str(ls)+"/log.err "+workingDir+"/"+str(ls)+"/launch.sh")
        
for job in njobs_list:
    print job






