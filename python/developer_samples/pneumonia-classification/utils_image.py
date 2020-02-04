import os
import matplotlib
import matplotlib.pyplot as plt

def show_results(job_id, hw):
    result_file="results/"+hw+"/result"+job_id+".txt"
    file_ready= os.path.isfile(result_file)
    if file_ready:
        count=0
        with open(result_file) as f:
            for ind, line in enumerate(f):
                if line=="\n":
                    break
                print(line)
                image_file='results/'+hw+'/result'+job_id+'_'+str(ind)+'.png'
                im=plt.imread(image_file)
                plt.figure(figsize = (20,20))
                plt.box(False)
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                plt.imshow(im)
                plt.show()
    else: 
        print("The results are not ready yet, please retry")
