class cnn_model;
    integer noOfKernels;              
    integer inputHeight, inputWidth;  
    integer kernelHeight, kernelWidth;
    integer stride;
    real inputMat[][];               
    real kernel[][][];               
    real outputMat[][][];
    integer outputWidth, outputHeight;             
    function new(input integer inputHeight, input integer inputWidth, 
                 input integer kernelHeight, input integer kernelWidth, 
                 input integer stride, input integer noOfKernels);
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        
        this.outputHeight = (inputHeight - kernelHeight) / stride + 1;
        this.outputWidth = (inputWidth - kernelWidth) / stride + 1;
        this.noOfKernels = noOfKernels; 
        this.inputMat = new[inputHeight];
        for (int i = 0; i < inputHeight; i++) begin
            this.inputMat[i] = new[inputWidth];
        end
        this.kernel = new[noOfKernels];
        for (int k = 0; k < noOfKernels; k++) begin
            this.kernel[k] = new[kernelHeight];
        for (int i = 0; i < kernelHeight; i++) begin
            this.kernel[k][i] = new[kernelWidth];
        end
        end
        this.outputMat  = new [noOfKernels];
        for (int k = 0; k < noOfKernels; k++) begin
            this.outputMat[k] = new[kernelHeight];
            for (int i = 0; i < outputHeight; i++) begin
                this.outputMat[k][i] = new[outputWidth];
        end
        end
    endfunction
    function void setInputAndKernel(input real inputMat[][], input real kernel[][][]);
        for (int i = 0; i < inputHeight; i++) begin
            for (int j = 0; j < inputWidth; j++) begin
                this.inputMat[i][j] = inputMat[i][j];
            end
        end
        for(int k = 0; k < noOfKernels ; k++) begin
        for (int i = 0; i < kernelHeight; i++) begin
            for (int j = 0; j < kernelWidth; j++) begin
                this.kernel[k][i][j] = kernel[k][i][j];
            end
        end
        end
    endfunction
    function setOut(input integer k , input real o[][]);
        for(int i = 0; i < outputHeight ; i ++) begin
            for (int j =0 ; j < outputWidth ;  j ++) begin
                this.ouputMat[k][i][j] = o[i][j];
            end
        end
    endfunction;
    function void apply();
        for(int k = 0; k < noOfKernels ; k++) begin
              Conv2D conv = new(inputHeight, inputWidth, kernelHeight, kernelWidth, stride);
              conv.setInputAndKernel(inputMat, kernel[k]);
              setOut(k,conv.outputMat);            
        end 
    endfunction
    
    function void displayOutput(input integer k);
        $display("Output Matrix:");
        for (int i = 0; i < outputHeight; i++) begin
            for (int j = 0; j < outputWidth; j++) begin
                $write("%0.2f ", outputMat[k][i][j]);
            end
            $display("");  
        end
    endfunction
endclass




// testbench
module test_conv2D;
    // Define parameters
    integer inputHeight = 3;
    integer inputWidth = 3;
    integer kernelHeight = 2;
    integer kernelWidth = 2;
    integer stride = 1;
    integer noOfKernels = 2; // Set the number of kernels to 2

    // Testbench variables
    real inputMat[3][3] = '{{1.0, 2.0, 3.0}, 
                           {4.0, 5.0, 6.0}, 
                           {7.0, 8.0, 9.0}};
    
    // Define two kernels
    real kernels[2][2][2] = '{{{1.0, 1.0}, 
                               {1.0, -1.0}}, // First kernel
                              {{0.0, 1.0}, 
                               {1.0, 0.0}}}; // Second kernel

    // Instantiate CNN model class
    cnn_model conv = new(inputHeight, inputWidth, kernelHeight, kernelWidth, stride, noOfKernels);

    initial begin
        // Set input matrix and kernels
        conv.setInputAndKernel(inputMat, kernels);

        // Apply the convolution
        conv.apply();

        // Display the output for each kernel
        for (int k = 0; k < noOfKernels; k++) begin
            conv.displayOutput(k);
        end
    end
endmodule

