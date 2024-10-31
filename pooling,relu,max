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


