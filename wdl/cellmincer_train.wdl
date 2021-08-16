version 1.0

workflow CellMincerTrain {

    call RunCellMincerTrain

    output {
        File model_state = RunCellMincerTrain.model_state
    }

}

task RunCellMincerTrain {

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
        Array[String] input_names
        Array[File] input_tar_gz
        File config
        File? pretrain
    }

    command <<<
        
        # extract CellMincer
        tar -xvzf ~{cellmincer_tar_gz}
        
        # install CellMincer
        pip install -e CellMincer/
        
        # extract all .tar.gz
        tar -xzvf ~{sep="; tar -xzvf " input_tar_gz}
            
        # run train
        cellmincer train -i ~{sep=" " input_names} -o . --config ~{config} ~{"--pretrain " + pretrain} --checkpoint checkpoint.tar.gz
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
         checkpointFile: "checkpoint.tar.gz"
    }

    output {
        File model_state = "model.pt"
    }
    
}
