# BD-Rate Evaluation

Bjøntegaard-Delta Rate method is widely used to evaluate the bitrate saving or coding efficiency for same quality (includes PSNR, SSIM, MS-SSIM and VMAF) between two different codecs. BD-rate calculates the average of the difference between the two RD curves corresponding to the two different coding methods. Specifically, it is necessary to curve-fit several (typically 4) points of the test to find the difference and take the average. In this project, the BD rate calculation for low, medium and high-quality ranges are measured by selecting the four lowest bitrates, four middle bitrates and four highest bitrates, respectively. 

In each time, four groups of bitrates and other values(PSNR,SSIM,MS-SSIM and VMAF) are input to the python file. 

Finally, all of the results are recorded in the file namely 'BD rate evaluation results'
