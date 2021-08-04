version 1.0

workflow CellMincerPreprocess {

    call RunCellMincerPreprocess

    output {
        File processed_tar_gz = RunCellMincerPreprocess.processed_tar_gz
    }

}

task RunCellMincerPreprocess {

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
        File input_file
        String input_name
        File manifest
        File config
        File? clean_reference
    }

    command <<<
    
        set -e
        
        # extract CellMincer
        tar -xvzf ~{cellmincer_tar_gz}
        
        # install CellMincer
        pip install -e CellMincer/

        # create output directory
        mkdir ~{input_name}
            
        # run preprocess
        cellmincer preprocess -i ~{input_file} -o ~{input_name} --manifest ~{manifest} --config ~{config} ~{"--clean " + clean_reference}

        # run feature
        cellmincer feature -i ~{input_name}/trend_subtracted.npy -o ~{input_name}/features.pkl

        tar --exclude="plots" --exclude=".*" -cvzf ~{input_name}.processed.tar.gz ~{input_name}
        
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
        File processed_tar_gz = "~{input_name}.processed.tar.gz"
    }
    
}
