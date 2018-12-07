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

 
def summaryPlot(results_dict, x_axis, y_axis, title):
    ''' Bar plot input:
	results_dict: dictionary of path to result file and label {path_to_result:label}
	x_axis: label of the x axis
	y_axis: label of the y axis
	title: title of the graph
    '''
    plt.figure()
    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    time = []
    arch = []
    diff = 0
    for path, hw in results_dict.items():
        if os.path.isfile(path):
            f = open(path, "r")
            label = round(float(f.readline()))
            time.append(label)
            f.close()
        else:
            time.append(-1)
        arch.append(hw)

    offset = max(time)/100
    for i in time:
        if i == -1:
            data = 'N/A'
            y = 0
        else:
            data = i
            y = i + offset   
        plt.text(diff, y, data, fontsize=10, multialignment="center",horizontalalignment="center", verticalalignment="bottom",  color='black')
        diff += 1
    plt.ylim(top=(max(time)+10*offset))
    plt.bar(arch, time, width=0.8, align='center')


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


   
def progressIndicator(path, title, min_, max_):
    '''
	Progress indicator reads first line in the file "path" 
	path: path to the progress file
	title: description of the bar
	min_: min_ value for the progress bar
	max_: max value in the progress bar

    '''
    style = {'description_width': 'initial'}
    progress_bar = widgets.FloatProgress(
    value=0,
    min=min_,
    max=max_,
    step=10,
    description=title,
    bar_style='info',
    orientation='horizontal',
    style=style
)
    remain_time = widgets.HTML(
    value='0',
    placeholder='0',
    description='Remaining:',
    style=style
)
    est_time = widgets.HTML(
    value='0',
    placeholder='0',
    description='Total Estimated:',
    style=style
)

    progress_bar.value=min_

    def _work(progress_bar, est_time,remain_time,  path):
        box_layout = widgets.Layout(display='flex', flex_flow='column', align_items='stretch', border='ridge', width='70%', height='')
        box = widgets.HBox([progress_bar, est_time, remain_time], layout=box_layout)
        display(box)
        # progress
        last_status = 0
        remain_val = '0'
        est_val = '0'
        output_file = path
        while last_status < 100:
            if os.path.isfile(output_file):
                with open(output_file, "r") as fh:
                    line1 = fh.readline() 	#Progress 
                    line2 = fh.readline()  	#Remaining time
                    line3 = fh.readline()  	#Estimated total time
                    if line1 and line2 and line3:
                        last_status = int(line1)
                        remain_val = int(line2)
                        est_val = int(line3)
                    progress_bar.value = last_status
                    remain_time.value = str(remain_val)+' seconds' 
                    est_time.value = str(est_val)+' seconds' 
            else:
                cmd = ['ls']
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                output,_ = p.communicate()
        remain_time.value = '0'+' seconds' 
        time.sleep(1)
        os.remove(output_file)


    thread = threading.Thread(target=_work, args=(progress_bar, est_time, remain_time, path))
    thread.start()

