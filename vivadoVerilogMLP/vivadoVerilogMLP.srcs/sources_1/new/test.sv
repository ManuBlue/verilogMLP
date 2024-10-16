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
                sum += prevLayerSummations[j] * weights[i][j]; 
            end
            sum += bias[i];
            outputs[i] = sum; 
        end
    endfunction
    
endclass
