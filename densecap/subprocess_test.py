from subprocess import Popen, PIPE

process = Popen(["th", "run_model.lua", "-input_image", "imgs/elephant.jpg", "-gpu", "-1"], stdout=PIPE)
stdout = process.communicate()
# print(stdout)

pass


