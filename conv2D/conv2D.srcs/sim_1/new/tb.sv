module test_conv2D;
    // Define parameters
    integer inputHeight = 3;
    integer inputWidth = 3;
    integer kernelHeight = 2;
    integer kernelWidth = 2;
    integer stride = 1;

    // Testbench variables
    real inputMat[3][3] = '{{1.0, 2.0, 3.0}, 
                           {4.0, 5.0, 6.0}, 
                           {7.0, 8.0, 9.0}};
    real kernel[2][2] = '{{1.0, 1.0}, 
                         {1.0, -1.0}};

    // Instantiate Conv2D class
    Conv2D conv = new(inputHeight, inputWidth, kernelHeight, kernelWidth, stride);

    initial begin
        // Set input matrix and kernel
        conv.setInputAndKernel(inputMat, kernel);

        // Apply the convolution
        conv.applyConvolution();

        // Display the output
        conv.displayOutput();
    end

endmodule
