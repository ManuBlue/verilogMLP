`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 31.10.2024 19:29:54
// Design Name: 
// Module Name: source
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
        //$display("Setting conv weights");
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
        $display("Applying 2D conv");
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

class featureMapGen;
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
            this.outputMat[k] = new[outputHeight];
            for (int i = 0; i < outputHeight; i++) begin
                this.outputMat[k][i] = new[outputWidth];
        end
        end
    endfunction
    function void setInputAndKernel(input real inputMat[][], input real kernel[][][]);
        //$display("Setting 3d weights");
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
//        $display("Size of outputmat:");
//        $display($size(outputMat));
//        $display($size(outputMat[0]));
//        $display($size(outputMat[0][0]));
        
        
        for(int i = 0; i < outputHeight ; i ++) begin
            for (int j =0 ; j < outputWidth ;  j ++) begin
                
                //$display("%0.2f ", o[i][j]);
                this.outputMat[k][i][j] = o[i][j];
            end
        end
    endfunction;
    function void apply();
        for(int k = 0; k < noOfKernels ; k++) begin
              Conv2D conv = new(inputHeight, inputWidth, kernelHeight, kernelWidth, stride);
              conv.setInputAndKernel(this.inputMat, this.kernel[k]);
              conv.applyConvolution();
              //$write("%0.2f ", inputMat[0][0]);
//              conv.displayOutput();
              setOut(k,conv.outputMat);            
        end 
    endfunction
    
    function void displayOutput(input integer k);
        $display("Output Matrix:");
        for (int i = 0; i < outputHeight; i++) begin
            for (int j = 0; j < outputWidth; j++) begin
                $write("%0.31f ", outputMat[k][i][j]);
            end
            $display("");  
        end
    endfunction
endclass






class max_pooling;
    
    integer poolSize;
    integer noOfInputLayers;
    integer inputHeight, inputWidth;
    real inputMat[][][];    
    real outputMat[][][];   
    integer outputWidth, outputHeight;

    function new(input integer inputHeight, input integer inputWidth, 
                 input integer poolSize, input integer noOfInputLayers);
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.poolSize = poolSize;
        this.noOfInputLayers = noOfInputLayers;
        this.outputWidth = inputWidth / poolSize;
        this.outputHeight = inputHeight / poolSize;

        this.inputMat = new[noOfInputLayers];
        this.outputMat = new[noOfInputLayers];

        for (int k = 0; k < noOfInputLayers; k++) begin
            this.inputMat[k] = new[inputHeight];
            this.outputMat[k] = new[outputHeight];
            
            for (int i = 0; i < inputHeight; i++) begin
                this.inputMat[k][i] = new[inputWidth];
            end
            
            for (int i = 0; i < outputHeight; i++) begin
                this.outputMat[k][i] = new[outputWidth];
            end
        end
    endfunction

    function void setInput(input real inputMat[][][]);
        for(int i = 0; i < noOfInputLayers; i++) begin
            for(int j = 0; j < inputHeight; j++) begin
                for(int k = 0; k < inputWidth; k++) begin
                    this.inputMat[i][j][k] = inputMat[i][j][k];
                end 
            end
        end
    endfunction

    function void apply();
        for (int layer = 0; layer < noOfInputLayers; layer++) begin
            for (int i = 0; i < outputHeight; i++) begin
                for (int j = 0; j < outputWidth; j++) begin
                    real maxVal = -1.0e30; 

                    for (int m = 0; m < poolSize; m++) begin
                        for (int n = 0; n < poolSize; n++) begin
                            int x = i * poolSize + m;
                            int y = j * poolSize + n;
                            if (x < inputHeight && y < inputWidth) begin
                                maxVal = (inputMat[layer][x][y] > maxVal) ? inputMat[layer][x][y] : maxVal;
                            end
                        end
                    end
                    outputMat[layer][i][j] = maxVal;
                end
            end
        end
    endfunction

function void display();
    for (int layer = 0; layer < noOfInputLayers; layer++) begin
        $display("Layer %0d:", layer);
        for (int i = 0; i < outputHeight; i++) begin
            for (int j = 0; j < outputWidth; j++) begin
                $write("%0f ", outputMat[layer][i][j]);
            end
            $display(""); 
        end
        $display(""); 
    end
endfunction


endclass



class ReLu;

    integer inputHeight, inputWidth;
    integer noOfInputLayers;
    real inputMat[][][];    
    real outputMat[][][];   

    function new(input integer inputHeight, input integer inputWidth, input integer noOfInputLayers);
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.noOfInputLayers = noOfInputLayers;

        this.inputMat = new[noOfInputLayers];
        this.outputMat = new[noOfInputLayers];

        for (int k = 0; k < noOfInputLayers; k++) begin
            this.inputMat[k] = new[inputHeight];
            this.outputMat[k] = new[inputHeight];
            
            for (int i = 0; i < inputHeight; i++) begin
                this.inputMat[k][i] = new[inputWidth];
                this.outputMat[k][i] = new[inputWidth];
            end
        end
    endfunction

    function void setInput(input real inputMat[][][]);
        for(int i = 0; i < noOfInputLayers; i++) begin
            for(int j = 0; j < inputHeight; j++) begin
                for(int k = 0; k < inputWidth; k++) begin
                    this.inputMat[i][j][k] = inputMat[i][j][k];
                end 
            end
        end
    endfunction

    function void apply();
        for (int layer = 0; layer < noOfInputLayers; layer++) begin
            for (int i = 0; i < inputHeight; i++) begin
                for (int j = 0; j < inputWidth; j++) begin
                    outputMat[layer][i][j] = (inputMat[layer][i][j] > 0) ? inputMat[layer][i][j] : 0.0;
                end
            end
        end
    endfunction

    function void display();
        for (int layer = 0; layer < noOfInputLayers; layer++) begin
            $display("Layer %0d (ReLU Output):", layer);
            for (int i = 0; i < inputHeight; i++) begin
                for (int j = 0; j < inputWidth; j++) begin
                    $write("%0f ", outputMat[layer][i][j]);
                end
                $display(""); 
            end
            $display(""); 
        end
    endfunction

endclass

class Flatten;

    integer inputHeight, inputWidth;
    integer noOfInputLayers;
    real inputMat[][][];    
    real outputMat[];       
    integer outputSize;     

    function new(input integer inputHeight, input integer inputWidth, input integer noOfInputLayers);
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.noOfInputLayers = noOfInputLayers;

        this.outputSize = noOfInputLayers * inputHeight * inputWidth;

        this.inputMat = new[noOfInputLayers];
        this.outputMat = new[outputSize];

        for (int k = 0; k < noOfInputLayers; k++) begin
            this.inputMat[k] = new[inputHeight];
            for (int i = 0; i < inputHeight; i++) begin
                this.inputMat[k][i] = new[inputWidth];
            end
        end
    endfunction

    function void setInput(input real inputMat[][][]);
        for (int i = 0; i < noOfInputLayers; i++) begin
            for (int j = 0; j < inputHeight; j++) begin
                for (int k = 0; k < inputWidth; k++) begin
                    this.inputMat[i][j][k] = inputMat[i][j][k];
                end 
            end
        end
    endfunction

    function void apply();
        integer index = 0;
        for (int layer = 0; layer < noOfInputLayers; layer++) begin
            for (int i = 0; i < inputHeight; i++) begin
                for (int j = 0; j < inputWidth; j++) begin
                    outputMat[index] = inputMat[layer][i][j];
                    index++;
                end
            end
        end
    endfunction

    function void display();
        $display("Flattened Output Vector:");
        for (int i = 0; i < outputSize; i++) begin
            $write("%0f ", outputMat[i]);
        end
        $display(""); 
    endfunction

endclass


class layer;
    integer units;
    integer prevLayerUnitCount;
    real weights[][];
    real bias[];
    real outputs[];
    
    function new(input integer units, input integer prevLayerUnitCount);
        this.units = units;
        this.prevLayerUnitCount = prevLayerUnitCount;

        this.weights = new[units];
        for (int i = 0; i < units; i++) begin
            this.weights[i] = new[prevLayerUnitCount];
        end

        this.bias = new[units];
        this.outputs = new[units];
    endfunction
    
    function setWeights(input real weights[][], input real bias[]);
        for (integer i = 0; i < units; i++) begin
            for (integer j = 0; j < prevLayerUnitCount; j++) begin
                this.weights[i][j] = weights[i][j];
            end
        end
        for (integer i = 0; i < units; i++) begin
            this.bias[i] = bias[i];
        end
    endfunction
    
    
    function void forward(input real prevLayerSummations[]);
        for (int i = 0; i < units; i++) begin
            real sum = 0;
            for (int j = 0; j < prevLayerUnitCount; j++) begin
                sum += prevLayerSummations[j] * this.weights[i][j]; 
            end
            sum += this.bias[i];
            this.outputs[i] = sum; 
        end
    endfunction
    function void display();
        $display("NN layer output:");
        for (int i = 0; i < this.units; i++) begin
            $write("%0.31f ", outputs[i]);
        end
        $display(""); 
    endfunction
    
endclass



class bias;
    real inputMat [][][];
    real bias [];
    real outputMat[][][];
    integer inputWidth,inputHeight,noOfInputLayers;
    function new(input integer inputHeight, input integer inputWidth, input integer noOfInputLayers);
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.noOfInputLayers = noOfInputLayers;
        this.inputMat = new[noOfInputLayers];
        this.outputMat = new[noOfInputLayers];
        this.bias  = new[noOfInputLayers];
        for (int k = 0; k < noOfInputLayers; k++) begin
            this.inputMat[k] = new[inputHeight];
            this.outputMat[k] = new [inputHeight];
            for (int i = 0; i < inputHeight; i++) begin
                this.inputMat[k][i] = new[inputWidth];
                this.outputMat[k][i] = new [inputWidth];
            end
        end
    endfunction
    function void setInputAndBias(input real inputMat[][][] , input real bias[]);
        for (int i = 0; i < noOfInputLayers; i++) begin
            for (int j = 0; j < inputHeight; j++) begin
                for (int k = 0; k < inputWidth; k++) begin
                    this.inputMat[i][j][k] = inputMat[i][j][k];
                    
                end 
            end
            
        end
        for(int i = 0 ; i < noOfInputLayers ; i ++ ) begin
            this.bias[i] = bias[i];
        end
    endfunction
    
    function void apply();
        for(int i = 0 ; i < noOfInputLayers ; i ++ ) begin
            for( int j = 0 ; j < inputHeight ; j ++ )begin
                for(int k = 0 ; k < inputWidth ; k++) begin 
                    this.outputMat[i][j][k] = this.inputMat[i][j][k] + this.bias[i];
                end
            end
        end
    endfunction
endclass

class ReLu1D;
    real size;
    real outputs[]; 
    function new(input integer size);
        this.size = size;
        this.outputs = new[size];
    endfunction
    function void apply(input real inputArr[]);
        for (int j = 0; j < this.size; j++) begin
            this.outputs[j] = (inputArr[j] > 0) ? inputArr[j] : 0.0;
        end
    endfunction
    
endclass

class softmax;
    real size;
    real outputs[]; 
    function new(input integer size);
        this.size = size;
        this.outputs = new[size];
    endfunction
    function void apply(input real inputArr[]);
        real total;
        total = 0;
        for (int j = 0; j < this.size; j++) begin
            total = total + $exp(inputArr[j]);
        end
        for (int j = 0; j < this.size; j++) begin
            this.outputs[j] = $exp(inputArr[j])/total;
        end
    endfunction
    function void display();
        $display("Softmax layer output:");
        for (int i = 0; i < this.size; i++) begin
            $write("%0.31f ", this.outputs[i] * 100,"percent");
        end
        $display(""); 
    endfunction
endclass
module test2(
);
endmodule
