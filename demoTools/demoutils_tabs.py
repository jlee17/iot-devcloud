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
    
        plt.figure(figsize=(15, 5))
        plt.title(title , fontsize=20, color='black', fontweight='bold')
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
    
       
    def progressIndicator(self, device, path, min_, max_, command):
        '''
    	Progress indicator reads first line in the file "path" 
    	path: path to the progress file
            file_name: file with data to track
    	title: description of the bar
    	min_: min_ value for the progress bar
    	max_: max value in the progress bar
    
        '''
        style = {'description_width': 'initial'}
        progress_bar_1 = widgets.FloatProgress(
            value=min_,
            min=min_,
            max=max_,
            description='',
            bar_style='info',
            orientation='horizontal',
            style=style
        )
        remain_time_1 = widgets.HTML(
            value='0',
            placeholder='0',
            description='Remaining:',
            style=style
        )
        est_time_1 = widgets.HTML(
            value='0',
            placeholder='0',
            description='Total Estimated:',
            style=style
        )
        progress_bar_2 = widgets.FloatProgress(
            value=min_,
            min=min_,
            max=max_,
            description='',
            bar_style='info',
            orientation='horizontal',
            style=style
        )
        remain_time_2 = widgets.HTML(
            value='0',
            placeholder='0',
            description='Remaining:',
            style=style
        )
        est_time_2 = widgets.HTML(
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
        file_name = [] 
        for name, title in self.prog_files:
            prog_file = '{path}/{name}{jobid}.txt'.format(path=path, name=name, jobid=self.jobDict[device]['jobid'])
            f = open(prog_file, "w")
            file_name.append(prog_file)
            f.close()
        
        def _work(progress_bar_1, progress_bar_2, remain_time_1, remain_time_2, est_time_1, est_time_2, op_display, op_video):
            box_layout = widgets.Layout(display='flex', flex_flow='column', align_items='stretch', border='ridge', width='100%', height='')
            frame_layout = widgets.Layout(display='flex', flex_flow='column', align_items='stretch', border='', width='100%', height='')
            widget_list = []
            title = widgets.HTML(value = '''<p>Job on {acc} in {edge}. Submission command:</p><pre>{command}</pre>'''.format(edge=device[0], acc=device[1], command=command))
            progress_bar_1.description = self.prog_files[0][1]
            if len(self.prog_files) > 1:
                progress_bar_2.description = self.prog_files[1][1]
                widget_list = [title, progress_bar_1, est_time_1, remain_time_1, progress_bar_2, est_time_2, remain_time_2, op_display, op_video]
            else:
                widget_list = [title, progress_bar_1, est_time_1, remain_time_1, op_display, op_video]

            if self.jobDict[device]['box_id'] == None:
                frame = widgets.HBox(widget_list, layout=frame_layout)
                cur_tabs = list(self.tab.children)
                cur_tabs.append(frame)
                self.tab.children = tuple(cur_tabs)
                self.tab.set_title(str(len(self.tab.children)-1),  '{target}'.format(target=device[1]))
                frame_id = len(self.tab.children)-1
                #self.jobDict[device]['box'] = frame 
                self.jobDict[device]['box_id'] = frame_id
                self.tab.selected_index = frame_id
            else:
                #frame = self.jobDict[device]['box'][0]
                frame = self.tab.children[self.jobDict[device]['box_id']]
                prev_frame = list(frame.children)
                for item in prev_frame:
                    item.close()
                frame.children = widget_list
                self.tab.selected_index = self.jobDict[device]['box_id']
            if not self.display_tab: 
                display(self.tab)
                self.display_tab = True
            # progress
            id_ = 1
            for output_file in file_name: 
                progress_bar =  widget_list[id_]
                est_time =  widget_list[id_+1]
                remain_time =  widget_list[id_+2]
                last_status = 0.0
                remain_val = '0'
                est_val = '0'
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
                id_ += 3
            op_display.disabled=False
            def wrapVideoHTML(event):
                op_video.value = self.videoHTML(path, device)
            op_display.on_click(wrapVideoHTML)

        thread = threading.Thread(target=_work, args=(progress_bar_1, progress_bar_2, remain_time_1, remain_time_2, est_time_1, est_time_2, op_display, op_video))
        thread.start()
        time.sleep(0.1)
        
     
    def displayHW(self):
        with open('~/Reference-samples/iot-devcloud/demoTools/database.json') as f:
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
        '''
           
        '''
        with open('~/Reference-samples/iot-devcloud/demoTools/database.json') as f:
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
                devlist.append("{target} in {edge}".format(edge=edge, target=target))
                dev_dict[index] = (edge, target)
                index += 1
        ##Display select list to submit job-----
        device = widgets.RadioButtons(
                 options=devlist,
                 rows=len(devlist),
                 description='',
                 disabled=False,
                 layout = Layout(width='100%')
                 )
        list_title = HTML('''<p>Choose target compute device and edge compute node:</p>''')
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
        status = widgets.HTML(value = '')
        def update_val(args):
            results.value = ('results/{edge}/{target}'.format(edge=dev_dict[args['new']][0], target=dev_dict[args['new']][1])).replace(" ","_") 
        

        device.observe(update_val, 'index')
        display(list_title)
        display(device)
        display(results)
        display(video)
        display(submit)
        display(status)
        
      
        def prepJob(event):
            edge, acc = dev_dict[device.index]
            #Check if previous job is running
            if (edge, acc) in self.jobDict.keys():
                if self.jobStillRunning(edge, acc):
                    msg =  "Could not submit a job to {acc} in {edge}: another job is still running".format(edge=edge, acc=acc)
                    status.value = msg
                    return
                else:
                    for x in self.jobDict[(edge, acc)]['msg']:
                        x.close()
            else:
                self.jobDict[(edge, acc)] = {}
                self.jobDict[(edge, acc)]['msg'] = []
                self.jobDict[(edge, acc)]['box_id'] = None
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
            status.value = "Job submitted to {acc} in {edge}".format(edge=edge, acc=acc)
            self.progressIndicator((edge, acc), results.value, 0, 100, command)
        submit.on_click(prepJob)
    
    
    def jobStillRunning (self, edge, accel):
        ''' Input: edge and target device
            Return: True if job still running, false if job terminated
        '''
        cmd = 'qstat '+self.jobDict[(edge, accel)]['jobid']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        output,_ = p.communicate()
        return output.decode("utf-8").rstrip() != ''
    
