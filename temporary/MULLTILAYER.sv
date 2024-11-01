`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/28/2024 02:23:53 PM
// Design Name: 
// Module Name: test
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
class Conv2D;
    // Parameters
    integer inputHeight, inputWidth;  // Dimensions of the input matrix
    integer kernelHeight, kernelWidth; // Dimensions of the kernel
    integer stride;  // Stride of convolution
    integer outputHeight, outputWidth; // Dimensions of the output matrix
    real inputMat[][];
    real kernel[][];
    real outputMat[][];

    // Constructor: Initialize the convolution layer
    function new(input integer inputHeight, input integer inputWidth, 
                 input integer kernelHeight, input integer kernelWidth, 
                 input integer stride);
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;

        // Calculate and store output size
        this.outputHeight = (inputHeight - kernelHeight) / stride + 1;
        this.outputWidth = (inputWidth - kernelWidth) / stride + 1;

        // Allocate memory for input, kernel, and output
        this.inputMat = new[inputHeight];
        for (int i = 0; i < inputHeight; i++) begin
            this.inputMat[i] = new[inputWidth];
        end

        this.kernel = new[kernelHeight];
        for (int i = 0; i < kernelHeight; i++) begin
            this.kernel[i] = new[kernelWidth];
        end

        this.outputMat = new[outputHeight];
        for (int i = 0; i < outputHeight; i++) begin
            this.outputMat[i] = new[outputWidth];
        end
    endfunction

    // Function to set input matrix and kernel values
    function void setInputAndKernel(input real inputMat[][], input real kernel[][]);
        // Set the input matrix
        for (int i = 0; i < inputHeight; i++) begin
            for (int j = 0; j < inputWidth; j++) begin
                this.inputMat[i][j] = inputMat[i][j];
            end
        end

        // Set the kernel
        for (int i = 0; i < kernelHeight; i++) begin
            for (int j = 0; j < kernelWidth; j++) begin
                this.kernel[i][j] = kernel[i][j];
            end
        end
    endfunction

    // Convolution function (similar to your Python implementation)
    function void applyConvolution();
        real sum;

        // Perform 2D convolution
        for (int i = 0; i < outputHeight; i++) begin
            for (int j = 0; j < outputWidth; j++) begin
                sum = 0;
                for (int ki = 0; ki < kernelHeight; ki++) begin
                    for (int kj = 0; kj < kernelWidth; kj++) begin
                        sum += inputMat[i * stride + ki][j * stride + kj] * kernel[ki][kj];
                    end
                end
                outputMat[i][j] = sum;
            end
        end
    endfunction

    // Function to display the output matrix
    function void displayOutput();
        $display("Output Matrix:");
        for (int i = 0; i < outputHeight; i++) begin
            for (int j = 0; j < outputWidth; j++) begin
                $write("%0.2f ", outputMat[i][j]);
            end
            $display("");  // New line
        end
    endfunction

endclass

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
    function void setOut(input integer k , input real o[][]);
        for(int i = 0; i < outputHeight ; i ++) begin
            for (int j =0 ; j < outputWidth ;  j ++) begin
                this.outputMat[k][i][j] = o[i][j];
            end
        end
    endfunction;
    function void apply();
        for(int k = 0; k < noOfKernels ; k++) begin
              Conv2D conv = new(inputHeight, inputWidth, kernelHeight, kernelWidth, stride);
              conv.setInputAndKernel(this.inputMat, this.kernel[k]);
              conv.applyConvolution();
              $write("%0.2f ", inputMat[0][0]);
//              conv.displayOutput();
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
    integer noOfKernels = 1; // Set the number of kernels to 2

    // Testbench variables
    real inputMat[3][3] = '{{1.0, 2.0, 3.0}, 
                           {4.0, 5.0, 6.0}, 
                           {7.0, 8.0, 9.0}};
    
    // Define two kernels
real kernels[1][2][2] = '{'{'{1.0, 1.0}, '{1.0, -1.0}}}; 

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
