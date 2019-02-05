from IPython.core.display import HTML
import ipywidgets as widgets
import subprocess

class DemoCatalog:

    def __init__(self):
        repoStatus = None
        statusButton = None
        self.repoStatus = widgets.HTML(value="(waiting to check the status; click the Check Status button to check immediately)")
        self.ShowRepositoryStatus()
        self.ShowRefreshButton()
        self.ShowListOfDemos()
        self.AutoClickStatusButton()

    def ShowRepositoryStatus(self):
        data = "<h2>Repository Status</h2>"
        display(HTML(data))
        self.statusButton = widgets.Button(description='Check Status')
        self.statusButton.on_click(self.RefreshStatus)
        display(self.repoStatus)
        display(self.statusButton)

    def AutoClickStatusButton(self):
        code = ('<script>autoClickLaunched = 0;'+
                'function AutoClickStatusButton(event) {'+
                '  var btns = document.getElementsByTagName("button");'+
                '  var text = "Check Status";'+
                '  for (var i = 0; i < btns.length; i++) {'+
                '    if (btns[i].textContent == text) {'+
                '       btns[i].click();'+
                '    }'+
                '  }'+
                '  if (!event) setTimeout(AutoClickStatusButton, 600000);'+
                '}'+
                'setTimeout(AutoClickStatusButton, 2000);'+
                '</script>')
        display(HTML(code))

    def ShowRefreshButton(self):
        data = "<h2>Refresh Button</h2>"
        display(HTML(data))

    def ShowListOfDemos(self):
        data = "<h2>List of Demos</h2>"
        display(HTML(data))

    def RefreshStatus(self, evt):
        cmd = "demoTools/checkStatus.sh"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output,_ = p.communicate()
        data = output.decode().split("\n")
        if int(data[1]) == 0:
            v = "the catalog is up to date. Feel free to start using the examples."
        elif int(data[1]) == 1:
            v = "there are important updates for the demos in the upstream repository. You should use the Refresh button below to get the most recent examples."
        elif int(data[1]) == 2:
            v = "it seems that you have started your own version control. If you ever want to revert your changes and start fresh, use the Refresh button below."
        else:
            v = "unable to determine status due to a server-side error."
        self.repoStatus.value = "<ul><li>Remote URL: "+data[0]+"</li><li>Time of last check: "+data[2]+"</li><li>Status: "+v
