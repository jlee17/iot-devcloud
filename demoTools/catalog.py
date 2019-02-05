from IPython.core.display import HTML, Markdown
import ipywidgets as widgets
import subprocess
import json
import os.path

class DemoCatalog:

    def __init__(self, config_file):
        with open(config_file, "r") as config:
            self.conf = json.load(config)
            config.close()
        with open(self.conf['css']) as css:
            display(HTML(css.read()))
        self.ShowRepositoryControls()
        self.ShowListOfDemos()

    def ShowRepositoryControls(self):
        url, status, lastCheck, fullstatus = self.GetStatus()
        msgs = self.conf['status']['messages']
        if int(status) == 0:
            c = 'uptodate'
        elif int(status) == 1:
            c = 'behind'
        elif int(status) == 2:
            c = 'ahead'
        else:
            c = 'unable'
        v = msgs[c].format(time=lastCheck)


        w_url = widgets.HTML(value=("{remote}: {remote_url}").format(remote=msgs['remote'], remote_url=url))
        w_time = widgets.HTML(value=("{time}: {lastCheck}").format(time=msgs['lastCheck'], lastCheck=lastCheck))
        w_git = widgets.HTML(value=("{gitsaid}: {gitline}").format(gitsaid=msgs['gitsaid'], gitline=fullstatus))
        w_hint = widgets.HTML(value=msgs['foreword'])
        w_refresh=widgets.Button(description=self.conf['status']['button'])
        w_info=widgets.VBox([w_url, w_time, w_git, w_hint, w_refresh])
        w_acc=widgets.Accordion(children=[w_info], selected_index=None)
        w_acc.set_title(0, v)
        w_acc.add_class(c)
        display(w_acc)
        
        self.refreshButton = w_refresh
        self.refreshButton.on_click(self.RefreshRepository)

    def ShowListOfDemos(self):
        data = "## "+self.conf['list']['header']+"\n"
        for lab in self.conf['list']['labs']:
            lab_dir=os.path.dirname(lab)
            with open(lab_dir+"/README.md", "r") as readme:
                cont=readme.read()
                readme.close()
                data += cont
            title = cont[0]
            data += "\n<a href='"+lab+"' target='_blank' class='big-jupyter-button'>"+self.conf['list']['messages']['goto']+": "+lab+"</a>\n"
        display(Markdown(data))

    def GetStatus(self):
        cmd = self.conf['status']['serverSideStatusScript']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output,_ = p.communicate()
        data = output.decode().split("\n")
        return data[0], data[1], data[2], data[3]
        
    def RefreshRepository(self, evt):
        self.refreshButton.disabled = True
        cmd = self.conf['status']['serverSideRefreshScript']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output,_ = p.communicate()
        display(HTML(self.conf['status']['reloadCode']))

        
