module tb_neural_network;

  // Declare layer object
  layer my_layer;
  
  // Input parameters for the layer
  real weights[2][3] = '{'{0.2, 0.4, 0.6}, '{0.1, 0.3, 0.5}};
  real biases[2] = '{0.5, 0.5};  
  real prevLayerSummations[3] = '{1.0, 2.0, 3.0}; // Inputs from previous layer
  
  initial begin
    // Instantiate the layer with 2 units and previous layer having 3 units
    my_layer = new(2, 3);
    
    // Set weights and biases
    my_layer.setWeights(weights, biases);

    // Perform the forward pass
    my_layer.forward(prevLayerSummations);

    // Display the outputs
    for (int i = 0; i < 2; i++) begin
      $display("Output[%0d] = %0f", i, my_layer.outputs[i]);
    end
  end
endmodule
