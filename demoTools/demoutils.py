from IPython.core.display import HTML
import threading
from IPython.display import display, Image
import ipywidgets as widgets
from ipywidgets import Layout
import time
import queue
import subprocess
import datetime
import matplotlib
import matplotlib.pyplot as plt
import os 
import warnings
import json


def progressUpdate(file_name, time_diff, frame_count, video_len):
        progress = round(100*(frame_count/video_len), 1)
        remaining_time = round((time_diff/frame_count)*(video_len-frame_count), 1)
        estimated_time = round((time_diff/frame_count)*video_len, 1)
        with  open(file_name, "w") as progress_file:
            progress_file.write(str(progress)+'\n')
            progress_file.write(str(remaining_time)+'\n')
            progress_file.write(str(estimated_time)+'\n')
    
class Demo:
    def __init__(self, jobscript, num_progress, video):
        self.jobDict = {}
        self.jobscript = jobscript
        self.prog_files = []
        self.prog_files.append(("i_progress_", "Inference"))
        self.video = video
        self.tab = widgets.Tab()
        self.tab.children = []
        self.display_tab = False
        if num_progress == 2:
            self.prog_files.append(("v_progress_", "Rendering"))
          

    def videoHTML(self, result_path, device):
        '''
           device: tuple of edge and accelerator
        '''
        videos_list = []
        #result_path = ('results/{edge}/{target}/'.format(edge=device[0], target=device[1])).replace(" ","_") 
        stats = result_path+'/stats.txt'
        for vid in os.listdir(result_path):
            if vid.endswith(".mp4"):
                videos_list.append(result_path+'/'+vid)
        if os.path.isfile(stats):
            with open(stats) as f:
                time = f.readline()
                frames = f.readline()
            stats_line = "<p>{frames} frames processed in {time} seconds</p>".format(frames=frames, time=time)
    
        else:
            stats_line = ""
        video_string = ""
        title = "Inference on {edge} with {accel}".format(edge=device[0], accel=device[1]) 
        height = '480' if len(videos_list) == 1 else '240'
        for x in range(len(videos_list)):
            video_string += "<video alt=\"\" controls autoplay height=\""+height+"\"><source src=\""+videos_list[x]+"\" type=\"video/mp4\" /></video>"
        output ='''<h2>{title}</h2>
        {stats_line}
        {videos}
        '''.format(title=title, videos=video_string, stats_line=stats_line)
        return output
    
    def summaryPlot22(self, x_axis, y_axis, title, plot):
        ''' Bar plot input:
    	x_axis: label of the x axis
    	y_axis: label of the y axis
    	title: title of the graph
        '''
        warnings.filterwarnings('ignore')
        if plot=='time':
            clr = 'xkcd:blue'
        else:
            clr = 'xkcd:azure'
    
        plt.figure(figsize=(15, 8))
        plt.title(title , fontsize=28, color='black', fontweight='bold')
        plt.ylabel(y_axis, fontsize=16, color=clr)
        plt.xlabel(x_axis, fontsize=16, color=clr)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    
        val = []
        arch = []
        diff = 0
        for edge, accel in self.jobDict.keys():
            path = os.path.join(self.jobDict[(edge, accel)]['output'], 'stats.txt')
            if os.path.isfile(path):
                f = open(path, "r")
                l1_time = float(f.readline())
                l2_count = float(f.readline())
                if plot=="time":
                    val.append(round(l1_time))
                else:
                    val.append(round(l2_count/l1_time))
                f.close()
            else:
                val.append(0)
            arch.append('{edge}+{accel}'.format(edge=edge, accel=accel))
    
        offset = max(val)/100
        for i in val:
            if i == 0:
                data = 'N/A'
                y = 0
            else:
                data = i
                y = i + offset   
            plt.text(diff, y, data, fontsize=14, multialignment="center",horizontalalignment="center", verticalalignment="bottom",  color='black')
            diff += 1
        plt.ylim(top=(max(val)+10*offset))
        plt.bar(arch, val, width=0.5, align='center', color=clr)
    
    def summaryPlot(self, x_axis, y_axis, title, plot):
        ''' Bar plot input:
    	x_axis: label of the x axis
    	y_axis: label of the y axis
    	title: title of the graph
        '''
        warnings.filterwarnings('ignore')
        if plot=='time':
            clr = 'xkcd:blue'
        else:
            clr = 'xkcd:azure'
    
        plt.figure(figsize=(15, 8))
        plt.title(title , fontsize=28, color='black', fontweight='bold')
        plt.ylabel(y_axis, fontsize=16, color=clr)
        plt.xlabel(x_axis, fontsize=16, color=clr)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    
        val = []
        arch = {}
        diff = 0
        for edge, accel in self.jobDict.keys():
            path = os.path.join(self.jobDict[(edge, accel)]['output'], 'stats.txt')
            if os.path.isfile(path) and not self.jobStillRunning(edge, accel):
                if not edge in arch.keys():
                    arch[edge] = {}
                    arch[edge]['val'] = []
                    arch[edge]['target'] = []
                f = open(path, "r")
                l1_time = float(f.readline())
                l2_count = float(f.readline())
                if plot=="time":
                    arch[edge]['val'].append(round(l1_time))
                else:
                    arch[edge]['val'].append(round(l2_count/l1_time))
                f.close()
                arch[edge]['target'].append('{accel}'.format(accel=accel).replace(" ", "\n", 2))
        if len(arch) != 0:
            # set offset
            max_val = []
            for dev in arch:
                max_val.append(max(arch[dev]['val']))
            offset = max(max_val)/100
            plt.ylim(top=(max(max_val)+10*offset))

            for dev in arch:
                for i in arch[dev]['val']:
                    y = i+offset
                    plt.text(diff, y, i, fontsize=14, multialignment="center",horizontalalignment="center", verticalalignment="bottom",  color='black')
                    diff += 1
                plt.bar(arch[dev]['target'], arch[dev]['val'], width=0.5, align='center', label = dev)
            plt.legend(list(arch.keys()))
        else:
            print("Results are not available yet") 
 
    def liveQstat(self):
        cmd = ['qstat']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output,_ = p.communicate()
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        qstat = widgets.Output(layout={'width': '100%', 'border': '1px solid gray'})
        stop_signal_q = queue.Queue()
    
        def _work(qstat,stop_signal_q):
            while stop_signal_q.empty():
                cmd = ['qstat']
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                output,_ = p.communicate()
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                qstat.append_stdout(now+'\n'+output.decode()+'\n\n\n')
                qstat.clear_output(wait=True)
                time.sleep(1.0)
            print('liveQstat stopped')
        thread = threading.Thread(target=_work, args=(qstat, stop_signal_q))
    
        thread.start()
        sb = widgets.Button(description='Stop')
        def _stop_qstat(evt):
            stop_signal_q.put(True)
        sb.on_click(_stop_qstat)
        display(qstat)
        display(sb)
    
       
    def progressIndicator(self, device, path, file_name, min_, max_, progress_id):
        '''
    	Progress indicator reads first line in the file "path" 
    	path: path to the progress file
            file_name: file with data to track
    	title: description of the bar
    	min_: min_ value for the progress bar
    	max_: max value in the progress bar
    
        '''
        style = {'description_width': 'initial'}
        progress_bar = widgets.FloatProgress(
            value=min_,
            min=min_,
            max=max_,
            description=self.prog_files[0][1],
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
        
        op_display = widgets.Button(
            description='Display Output',
            disabled=True,
            button_style='',
            icon='check' 
        )
        op_video = widgets.HTML(
            value='',
            placeholder='',
            description='',
            style=style

        )
        #Check if results directory exists, if not create it and create the progress data file 
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        f = open(path+'/'+file_name, "w")
        f.close()
        
        def _work(progress_bar, remain_time , est_time, op_display, op_video):
            box_layout = widgets.Layout(display='flex', flex_flow='column', align_items='stretch', border='ridge', width='100%', height='')
            if progress_id == len(self.prog_files):
                box = widgets.HBox([progress_bar, est_time, remain_time, op_display, op_video], layout=box_layout)
            else:
                box = widgets.HBox([progress_bar, est_time, remain_time], layout=box_layout)
            self.jobDict[device]['box'].append(box)
            display(box)
            # progress
            last_status = 0.0
            remain_val = '0'
            est_val = '0'
            output_file = '{path}/{file_name}{jobid}.txt'.format(path=path, file_name=file_name, jobid=self.jobDict[device]['jobid']) 
            while last_status < 100:
                if os.path.isfile(output_file):
                    with open(output_file, "r") as fh:
                        line1 = fh.readline() 	#Progress 
                        line2 = fh.readline()  	#Remaining time
                        line3 = fh.readline()  	#Estimated total time
                        if line1 and line2 and line3:
                            last_status = float(line1)
                            remain_val = line2
                            est_val = line3
                        progress_bar.value = last_status
                        remain_time.value = remain_val+' seconds' 
                        est_time.value = est_val+' seconds' 
                else:
                    cmd = ['ls']
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    output,_ = p.communicate()
            remain_time.value = '0'+' seconds' 
            os.remove(output_file)
            op_display.disabled=False
            def wrapVideoHTML(event):
                op_video.value = self.videoHTML(path, device)
            op_display.on_click(wrapVideoHTML)

        thread = threading.Thread(target=_work, args=(progress_bar, remain_time, est_time, op_display, op_video))
        thread.start()
        time.sleep(0.1)
        
     
    def displayHW(self):
        with open('/home/u20675/Reference-samples/iot-devcloud/demoTools/database.json') as f:
            data = json.load(f)
        devlist = []
        hw = data["devices"]
        hw_list = "<ul>"
        for dev in hw:
            key = list(dev.keys())[0]
            data = dev[key]
            edge = data["edge"]
            proc = data["processor"]
            gpu = data["graphics"]
            acc = data["accelerator"]
            hw_list += "<li>{edge} edge node with {proc} processor, {gpu} ".format(edge=edge, proc=proc, gpu=gpu)
            if acc != "None":
                hw_list += "and {acc}".format(acc=acc)
            hw_list += "</li>"
        hw_list += "</ul>"
        #Display hw list-----
        display(HTML('''<h2> List of Available Architectures</h2> {hw_list}'''.format(hw_list=hw_list)))
    
    
    def jobSetup(self):
        with open('/home/u20675/Reference-samples/iot-devcloud/demoTools/database.json') as f:
            data = json.load(f)
        devlist = []
        done = []
        hw = data["devices"]
        accelerator = data["processing"]
        code = data["code"]
        index = 0
        dev_dict = {}
        for dev in hw:
            x = []
            key = list(dev.keys())[0]
            val = dev[key]
            edge = val["edge"]
            proc = val["processor"]
            gpu = val["graphics"]
            acc = val["accelerator"]
            if not (edge, proc, gpu) in done:
                x.append((edge, proc))
                x.append((edge, gpu))
                done.append((edge, proc, gpu))
            if acc != "None": 
                x.append((edge, acc))
            for edge, target in x:
                devlist.append("{edge} + {target}".format(edge=edge, target=target))
                dev_dict[index] = (edge, target)
                index += 1
        ##Display select list to submit job-----
        device = widgets.RadioButtons(
                 options=devlist,
                 rows=len(devlist),
                 description='Available devices',
                 disabled=False,
                 layout = Layout(width='100%')
                 )
        results = widgets.Text(
                  value=('results/{edge}/{target}'.format(edge=dev_dict[device.index][0], target=dev_dict[device.index][1])).replace(" ","_"),
                  placeholder='Type something',
                  description='Results Directory:',
                  layout = Layout(width='100%'),
                  disabled=False
                 )
        video = widgets.Text(
                value=self.video,
                placeholder='Type something',
                description='Input Video:',
                layout = Layout(width='100%'),
                disabled=False
                 )
        submit = widgets.Button(
                 description='Submit Job',
                 disabled=False,
                 button_style='',
                 icon='check' 
                 )

        def update_val(args):
            results.value = ('results/{edge}/{target}'.format(edge=dev_dict[args['new']][0], target=dev_dict[args['new']][1])).replace(" ","_") 
        
        device.observe(update_val, 'index')
        display(device)
        display(results)
        display(video)
        display(submit)
      
        def prepJob(event):
            edge, acc = dev_dict[device.index]
            #Check if previous job is running
            if (edge, acc) in self.jobDict.keys():
                if self.jobStillRunning(edge, acc):
                    msg = widgets.HTML(value = "<h3> Another job submitted to {edge} with {acc} is still running</h3>".format(edge=edge, acc=acc))
                    self.jobDict[(edge, acc)]['msg'].append(msg)
                    display(msg)
                    return
                else:
                    self.jobDict[(edge, acc)]['title'].close()
                    for x in self.jobDict[(edge, acc)]['box']:
                        x.close()
                    for x in self.jobDict[(edge, acc)]['msg']:
                        x.close()
            else:
                self.jobDict[(edge, acc)] = {}
                self.jobDict[(edge, acc)]['box'] = []
                self.jobDict[(edge, acc)]['msg'] = []
            node_e = code[edge]
            node_a = code[acc]
            dFlag = accelerator[acc]["-d"]
            FP = accelerator[acc]["FP"]
            command = "qsub {script} -l nodes=1:{node_e}:{node_a} -F \" {r_dir} {dFlag} {FP} {in_video} \" ".format(script=self.jobscript, node_e=node_e, node_a=node_a, r_dir=results.value, dFlag=dFlag, FP=FP, in_video=video.value)
            p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            output,_ = p.communicate()
            jobid = output.decode("utf-8").rstrip()
            self.jobDict[(edge, acc)]['jobid'] = jobid 
            self.jobDict[(edge, acc)]['output'] = results.value
            title = widgets.HTML(value = '''<h2> Job submitted to {edge} with computations running on {acc} </h2>'''.format(edge=edge, acc=acc))
            self.jobDict[(edge, acc)]['title'] = title 
            display(title)
            prog_id = 1;
            for file_name, title in self.prog_files:
                self.progressIndicator((edge, acc), results.value, file_name, 0, 100, prog_id)
                prog_id +=1
        submit.on_click(prepJob)
    
    
    def jobStillRunning (self, edge, accel):
        ''' Input: edge and target device
            Return: True if job still running, false if job terminated
        '''
        cmd = 'qstat '+self.jobDict[(edge, accel)]['jobid']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        output,_ = p.communicate()
        return output.decode("utf-8").rstrip() != ''
    
