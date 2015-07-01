% By: Behrang Mehrparvar
% based on paper: [Visualizing Higher-Layer Features of a Deep Network]
% 7/1/2015

function visualize_opt( sae, layer, lambda, lag, iter, mask, channels )
%This function visualizes the activation of hidden units in a higher layer
%of the autoencoder by maximizing the activation function

% sae: the input model
% layer: the layer you want to visualize
% lambda: learning rate
% lag: lagrange multiplier
% iter: number of iterations
% mask: list of hidden units you want to visualize. leave empty array for all
% channels: number of channels in input e.g. 3 for RGB (designed for CIFAR dataset)


%note: each hidden unit should be optimized separately (count)


    
    count = size(sae.ae{layer}.W{1},1);
    temp = unifrnd(0,1,1,sae.ae{1}.size(1,1));
    %temp = unifrnd(0,1,size(x));
    %temp = temp(:,1:1024);
    out = repmat(temp',1,count);
    
    for it = 1:iter
        for hid = 1:count
            h = out(:,hid)';          
            der = 1;
            for i = 1:layer
                m = size(h, 1);
%                 z = sae.ae{i}.W{1} * [ones(m,1) h]';

                nn2 = nnff(sae.ae{i}, h, h);
                h = nn2.a{2};
                h = h(:,2:end);
                
                switch sae.ae{i}.activation_function 
                    case 'sigm'
                        d_act = h .* (1 - h);
                    case 'tanh_opt'
                        d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * h.^2);
                    case 'softplus'
                        d_act = 1 ./ (1 + exp(-h));
                end
                d_act_diag = diag(d_act);
                W = sae.ae{i}.W{1};
                W = W(:,2:end);
                d = W' * d_act_diag;
                                
                der = der * d;

            end          
            deriv = der(:,hid) - 2 * lag * out(:,hid);
            out(:,hid) = out(:,hid) + lambda * deriv;
        
        end
    end
    ch_size = size(out,1)/channels;
    for ch = 1:1 %channels  % just visualize first channel
        st = (ch - 1) * ch_size + 1;
        en = ch * ch_size;
        out2 = out(st:en,:);
        figure('name',strcat('ch:',num2str(ch),'-','it: ',num2str(it),'-' , 'L: ',num2str(layer)));
        if(size(mask)>0)
            final = out2(:,mask);
        else
            final = out2;
        end
        visualize(final);   
    end
end

