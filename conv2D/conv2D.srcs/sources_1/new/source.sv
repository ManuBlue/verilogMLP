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
