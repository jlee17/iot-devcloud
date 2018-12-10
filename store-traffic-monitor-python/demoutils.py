from IPython.core.display import HTML
import threading
from IPython.display import display
import ipywidgets as widgets
import time
import queue
import subprocess
import datetime

def videoHTML(title, video_dir,stats=None):
    if stats:
        with open(stats) as f:
            fps = f.readline()
            
        stats_line = "<p>Average frame rate : {fps} fps</p>".format(fps=fps)
    else:
        stats_line = ""
    return HTML('''<h2>{title}</h2>
        {stats_line}
<video alt="" controls autoplay height='240'>
   <source src="{video_dir}/inference_output_Video_0.mp4" type="video/mp4" />
</video>
<video alt="" controls autoplay height='240'>
   <source src="{video_dir}/inference_output_Video_1.mp4" type="video/mp4" />
</video>
<video alt="" controls autoplay height='240'>
   <source src="{video_dir}/inference_output_Video_2.mp4" type="video/mp4" />
</video>
<video alt="" controls autoplay height='240'>
   <source src="{video_dir}/inference_output_stats.mp4" type="video/mp4" />
</video>'''.format(title=title, video_dir=video_dir, stats_line=stats_line))

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