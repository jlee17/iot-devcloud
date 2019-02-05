from IPython.core.display import HTML
import ipywidgets as widgets
import subprocess
import json

class DemoCatalog:

    def __init__(self, config_file):
        with open(config_file) as config:
            self.conf = json.load(config)

        self.repoStatus = widgets.HTML(value=self.conf['status']['messages']['placeholder'])
        self.ShowRepositoryStatus()
        self.ShowRefreshButton()
        self.ShowListOfDemos()
        self.AutoClickStatusButton()

    def ShowRepositoryStatus(self):
        data = "<h2>"+self.conf['status']['header']+"</h2>"
        display(HTML(data))
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
        data = "<h2>"+self.conf['refresh']['header']+"</h2>"
        data += "<p>"+self.conf['refresh']['foreword']+"<p>"
        self.refreshButton = widgets.Button(description=self.conf['refresh']['button'])
        self.refreshButton.on_click(self.RefreshRepository)
        display(HTML(data))
        display(self.refreshButton)

    def ShowListOfDemos(self):
        data = "<h2>"+self.conf['list']['header']+"</h2>"
        display(HTML(data))

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
        self.repoStatus.value = ("<ul><li>{remote}: {remote_url}</li>"+
                "<li>{time}: {time_last}</li>"+
                "<li>{status}: {status_value}</li></ul>").format(
                    remote=terms['remote'], remote_url=data[0],
                    time=terms['lastCheck'], time_last=data[2],
                    status=terms['status'], status_value=v)


    def RefreshRepository(self, evt):
        self.refreshButton.disabled = True
        cmd = self.conf['refresh']['serverSideScript']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output,_ = p.communicate()
        data = output.decode().split("\n")
        print(data)
        refreshCode="<script>window.location.reload()</script>"
        display(HTML(refreshCode))

        
