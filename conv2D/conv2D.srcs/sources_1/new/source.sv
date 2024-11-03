class Conv2D;
    
    integer inputHeight, inputWidth;  
    integer kernelHeight, kernelWidth; 
    integer stride;  
    integer outputHeight, outputWidth; 
    real inputMat[][];
    real kernel[][];
    real outputMat[][];

    
    function new(input integer inputHeight, input integer inputWidth, 
                 input integer kernelHeight, input integer kernelWidth, 
                 input integer stride);
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;

        
        this.outputHeight = (inputHeight - kernelHeight) / stride + 1;
        this.outputWidth = (inputWidth - kernelWidth) / stride + 1;

        
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

    
    function void setInputAndKernel(input real inputMat[][], input real kernel[][]);
        
        for (int i = 0; i < inputHeight; i++) begin
            for (int j = 0; j < inputWidth; j++) begin
                this.inputMat[i][j] = inputMat[i][j];
            end
        end

        
        for (int i = 0; i < kernelHeight; i++) begin
            for (int j = 0; j < kernelWidth; j++) begin
                this.kernel[i][j] = kernel[i][j];
            end
        end
    endfunction

    
    function void applyConvolution();
        real sum;

        
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

    
    function void displayOutput();
        $display("Output Matrix:");
        for (int i = 0; i < outputHeight; i++) begin
            for (int j = 0; j < outputWidth; j++) begin
                $write("%0.2f ", outputMat[i][j]);
            end
            $display("");  
        end
    endfunction

endclass
