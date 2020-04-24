# The whole process of this project
## 1. Overview
This project investigates the performance comparison of H.264/MPEG-AVC, H.265/MPEG-HEVC and VP9 encoders supported by FFmpeg for gaming content. Five diverse types of gaming content within 10 seconds are selected which include CSGO, DOTA2, RUST, STARCRAFT and WITCHER3. Eight different QP values are selected for x264, x265 and libvpx encoders, while these encoders are set to two configurations namely PT and VQ model. The PT model optimizes the PSNR and the VQ model aims to acquire high quality to be analysed by VMAF. The Bjøntegaard-Delta Rate method is used to evaluate the bitrate saving or coding efficiency. 

Basically, a common multimedia framework namely FFmpeg is an open-source project for handling different types of multimedia streams and content. It can support multiple video formats and is used for multiple other software projects, which can be compiled in different platforms, including Linux, Windows and macOS. Due to its universality, compatibility and multifunction, the encoders selected in this paper are all supported by FFmpeg. Furthermore, the whole project is executed on Windows platform by using python 3.7. 
## 2. Process
### Step 1: Download FFmpeg source code
In order to execute the FFmpeg command, the source code should be download from the following website:
https://ffmpeg.zeranoe.com/builds/
It provides support on Windows 64-bits, Window 32-bits and macOS 64-bit. The author chooses the builds of Windows 64-bits. 
After download and unzip the package, it contains files namely bin, doc, presets, LICENSE and README. Then open the ‘bin’  file, and create four new files namely ‘source-video’, ‘encode-video’, ‘vmaf’ and ‘video-format’ for future use. 
### Step 2: Download source video
Free copyright videos can be downloaded from the following link:
https://media.xiph.org/video/derf/
The author selected 5 gaming videos from gaming content. 
### Step 3: Bit-rate settings
In the encoding process, it is important to consider the trade-off between the quality of content and the rate control method. 
For both x264 and x265 encoder, the Quantization Parameter (QP) values are chosen from the following set:
QP(x264)=QP(x265)={15,19,23,27,3,35,39,43}
For libvpx encoder, the QP values are chosen from the following set:
QP(libvpx-VP9)={14,21,28,35,42,49,56,63}
### Step 4: Parameter settings of codecs.
It is important to set the parameters of each encoder before the experiment. Meanwhile, the meaning of each parameter should be understood clearly. 
The setting of prarmeters for auther are illustrated in file namely 'Patameter setting of x264 x265 and libvpx encoder'.Two models namely PSNR-tuned (PT) model and Visual Quality-tuned (VQ) model are set for each encoder to execute:
•	PT model: In this configuration, the options of psychovisual and the adaptive quantization are disabled, while the results of PSNR are optimized. 
•	VQ model: By contrast, the options of psychovisual and adaptive quantization are adopted which aims to optimize the visual quality. 
