from IPython.core.display import HTML
import threading
from IPython.display import display, Image
import ipywidgets as widgets
import time
import queue
import subprocess
import datetime
import matplotlib
import matplotlib.pyplot as plt
import os 


def videoHTML(title, video, stats=None):
    if stats:
        with open(stats) as f:
            time = f.readline()
            frames = f.readline()
        stats_line = "<p>{frames} frames processed in {time} seconds</p>".format(frames=frames, time=time)
    else:
        stats_line = ""
    return HTML('''<h2>{title}</h2>
    {stats_line}
<video alt="" controls autoplay height='480'>
   <source src="{video}" type="video/mp4" />
</video>
'''.format(title=title, video=video, stats_line=stats_line))



def summaryPlot(result_dir, HW):
    plt.figure()
    plt.title("Inference engine processing time")
    plt.ylabel("Time, seconds")
    time = []
    arch = []
    for hw in HW:
        path = os.path.join(result_dir, hw , 'stats.txt')
        f = open(path, "r")
        time.append(float(f.readline()))
        arch.append(hw)
        f.close()
    plt.bar(arch, time)
 


def liveQstat():
    cmd = ['qstat']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output,_ = p.communicate()
    now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    qstat = widgets.Textarea(value=now+'\n'+output.decode(), layout={'height': '200px', 'width': '600px'})
    stop_signal_q=queue.Queue()

    def _work(qstat,stop_signal_q):
        while stop_signal_q.empty():
            cmd = ['qstat']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output,_ = p.communicate()
            now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            qstat.value = now+'\n'+output.decode()
            time.sleep(0.1)
        print('qstat stopped')
    thread = threading.Thread(target=_work, args=(qstat,stop_signal_q))

    thread.start()
    sb = widgets.Button(description='Stop')
    def _stop_qstat(evt):
        stop_signal_q.put(True)
    sb.on_click(_stop_qstat)
    display(qstat)
    display(sb)


   
   	 
def inferProgress(fname, job_id):
    infer_progress = widgets.FloatProgress(
        value=0,
        min=0,
        max=100.0,
        step=1,
        description='Inference',
        bar_style='info',
        orientation='horizontal'
    )
    video_progress = widgets.FloatProgress(
        value=0,
        min=0,
        max=100.0,
        step=1,
        description='Post processing',
        bar_style='info',
        orientation='horizontal'
    )

    infer_progress.value=0
    video_progress.value=0
    display(infer_progress)
    display(video_progress)

    def _work(infer_progress, video_progress, fname, job_id):

        # Inference engine progress
        last_status=0
        infer_prog = os.path.join(fname, 'i_progress_'+job_id[0]+'.txt')
        while last_status < 100:
            if os.path.isfile(infer_prog):
                with open(infer_prog, "r") as fh:
                    line=fh.readline()
                    if line:
                        last_status = int(line)
                    infer_progress.value=last_status
            else:
                cmd = ['ls']
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                output,_ = p.communicate()
        time.sleep(1)
        os.remove(infer_prog)

        #Post processing progress
        last_status=0
        video_prog = os.path.join(fname, 'v_progress_'+job_id[0]+'.txt')
        while last_status < 100:
            if os.path.isfile(video_prog):
                with open(video_prog, "r") as fh:
                    line=fh.readline()
                    if line:
                        last_status = int(line)
                    video_progress.value=last_status
            else:
                cmd = ['ls']
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                output,_ = p.communicate()
        time.sleep(1)
        os.remove(video_prog)

    thread = threading.Thread(target=_work, args=(infer_progress, video_progress, fname, job_id))
    thread.start()



        
