import os
import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import deeplabcut


model_options = deeplabcut.create_project.modelzoo.Modeloptions
video_filenames = []
with open("filenames") as videofnames:
	video_filenames = [x.strip() for x in videofnames.readlines()]

for video_path in video_filenames:
	# video_path = "原因不明の喧嘩Rb_Mo.wmv"
	print (video_path)
	ProjectFolderName = 'monkey_'+video_path[:-4]
	YourName = 'rae'
	model2use = model_options[3]
	videotype = video_path[-3:]
	deeplabcut.DownSampleVideo(video_path, width=300)

	video_path=os.path.join(str(Path(video_path).stem)+'downsampled.'+videotype)
	path_config_file = deeplabcut.create_pretrained_project(ProjectFolderName, YourName, video_path, videotype=videotype, model=model2use, analyzevideo=True, createlabeledvideo=True, copy_videos=True) #must leave copy_videos=True