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
        self.ShowRepositoryStatus()
        self.ShowRefreshButton()
        self.ShowListOfDemos()
        self.AutoClickStatusButton()

    def ShowRepositoryControls(self):
        self.repoStatus = widgets.HTML(value=self.conf['status']['messages']['placeholder'])
        self.statusButton = widgets.Button(description=self.conf['status']['button'])
        
        display(Markdown("## Code Repository Controls"))
        display(widgets.VBox([self.repoStatus, self.statusButton]))

    def ShowRepositoryStatus(self):
        self.repoStatus = widgets.HTML(value=self.conf['status']['messages']['placeholder'])
        data = "### "+self.conf['status']['header']
        display(Markdown(data))
        self.statusButton = widgets.Button(description=self.conf['status']['button'])
        self.statusButton.on_click(self.RefreshStatus)
        display(self.repoStatus)
        display(self.statusButton)
        
    def AutoClickStatusButton(self):
        code = ('<script>autoClickLaunched = 0;'+
                'function AutoClickStatusButton(event) {'+
                '  var btns = document.getElementsByTagName("button");'+
                '  var text = "'+self.conf["status"]["button"]+'";'+
                '  for (var i = 0; i < btns.length; i++) {'+
                '    if (btns[i].textContent == text) {'+
                '       btns[i].click();'+
                '    }'+
                '  }'+
                '  if (!event) setTimeout(AutoClickStatusButton, '+self.conf['status']['autoCheckIntervalMilliseconds']+');'+
                '}'+
                'setTimeout(AutoClickStatusButton, '+self.conf['status']['firstCheckDelayMilliseconds']+');'+
                '</script>')
        display(HTML(code))

    def ShowRefreshButton(self):
        data = "### "+self.conf['refresh']['header']+"\n"
        data += self.conf['refresh']['foreword']
        self.refreshButton = widgets.Button(description=self.conf['refresh']['button'])
        self.refreshButton.on_click(self.RefreshRepository)
        display(Markdown(data))
        display(self.refreshButton)

    def ShowListOfDemos(self):
        data = "## "+self.conf['list']['header']+"\n"
        for lab in self.conf['list']['labs']:
            lab_dir=os.path.dirname(lab)
            with open(lab_dir+"/README.md", "r") as readme:
                cont=readme.read()
                readme.close()
                data += cont
            title = cont[0]
            data += "\n<a href='"+lab+"' target='_blank' class='big-jupyter-button'>"+self.conf['list']['terms']['goto']+": "+lab+"</a>\n"
        display(Markdown(data))

    def RefreshStatus(self, evt):
        cmd = self.conf['status']['serverSideScript']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output,_ = p.communicate()
        data = output.decode().split("\n")
        msgs = self.conf['status']['messages']
        if int(data[1]) == 0:
            v = msgs['upToDate']
        elif int(data[1]) == 1:
            v = msgs['behind']
        elif int(data[1]) == 2:
            v = msgs['ahead']
        else:
            v = msgs['unable']

        terms = self.conf['status']['terms']
        status = ("<ul><li>{remote}: {remote_url}</li>"+
                "<li>{time}: {time_last}</li>"+
                "<li>{status}: {status_value}</li></ul>").format(
                    remote=terms['remote'], remote_url=data[0],
                    time=terms['lastCheck'], time_last=data[2],
                    status=terms['status'], status_value=v)
        self.repoStatus.value = status

    def RefreshRepository(self, evt):
        self.refreshButton.disabled = True
        cmd = self.conf['refresh']['serverSideScript']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output,_ = p.communicate()
        data = output.decode().split("\n")
        print(data)
        refreshMessage="** Finished refresh. Please reload the browser window. **"
        display(Markdown(refreshMessage))

        
