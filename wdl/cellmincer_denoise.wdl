version 1.0

workflow CellMincerDenoise {

    call RunCellMincerDenoise

    output {
        File denoised = RunCellMincerDenoise.denoised
        File? avi = RunCellMincerDenoise.avi
        File? psnr = RunCellMincerDenoise.psnr
    }

}

task RunCellMincerDenoise {

    input {
        # runtime
        String docker_image
        Int hardware_boot_disk_size
        Int hardware_disk_size
        Int hardware_memory
        Int hardware_cpu_count
        String hardware_zones
        String hardware_gpu_type
        Int hardware_preemptible_tries
        
        # preprocess and feature
        File cellmincer_tar_gz
        String input_name
        File input_tar_gz
        File model
        File config
    }

    command <<<
        
        # extract CellMincer
        tar -xvzf ~{cellmincer_tar_gz}
        
        # install CellMincer
        pip install -e CellMincer/
        
        # extract all .tar.gz
        tar -xzvf input_tar_gz
            
        # run denoise
        cellmincer denoise -i ~{input_name} -o . --model ~{model} --config ~{config}
    >>>
    
    runtime {
         docker: "${docker_image}"
         bootDiskSizeGb: hardware_boot_disk_size
         disks: "local-disk ${hardware_disk_size} HDD"
         memory: "${hardware_memory}G"
         cpu: hardware_cpu_count
         zones: "${hardware_zones}"
         gpuCount: 1
         gpuType: "${hardware_gpu_type}"
         maxRetries: 10
         preemptible: hardware_preemptible_tries
    }

    output {
        File denoised = "denoised_tyx.npy"
        File? avi = "denoised.avi"
        File? psnr = "psnr_t.npy"
    }
    
}
