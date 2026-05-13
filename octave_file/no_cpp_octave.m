function out_struct = compute_ripple(name, data, include_mode)

    if nargin < 3
        include_mode = true;
    end

    name = char(name); 

    data_avg = mean(data);
    delta_i = max(data) - min(data);
    ripple_rate = delta_i / data_avg;

    if any(data < 0)
        mode = 'DCM';
    else
        mode = 'CCM';
    end

    out_struct = struct();
    out_struct.([name '_avg']) = data_avg;
    out_struct.(['delta_' name]) = delta_i;
    out_struct.([name '_ripple_rate']) = ripple_rate;

    if include_mode
        out_struct.([name '_Conduction_Mode']) = mode;
    end
end

try
    
    touch_time = 1;

    before_touch_idx = find(result.Time >= touch_time, 1) - 1;
    slice = floor(before_touch_idx * 0.7);

    L1 = result.Values(1,slice:before_touch_idx);
    L1_struct = compute_ripple("i_L1", L1);

    L2 = result.Values(2,slice:before_touch_idx);
    L2_struct = compute_ripple("i_L2", L2);


    Vout = result.Values(3,slice:before_touch_idx);
    Vout_struct = compute_ripple("V_out", Vout, false);


    C1 = result.Values(4,slice:before_touch_idx);
    C1_struct = compute_ripple("v_C1", C1, false);

    mosfet_cond_loss = mean(result.Values(5, slice:before_touch_idx));
    mosfet_switch_loss = mean(result.Values(6, slice:before_touch_idx));
    diode_cond_loss = mean(result.Values(7, slice:before_touch_idx));


    L1_ESR_loss = mean(result.Values(8, slice:before_touch_idx));
    L2_ESR_loss = mean(result.Values(9, slice:before_touch_idx));
    C1_ESR_loss = mean(result.Values(10, slice:before_touch_idx));
    C2_ESR_loss = mean(result.Values(11, slice:before_touch_idx));

    efficiency = max(result.Values(12, slice:before_touch_idx));

    loss_struct = struct( ...
                "mosfet_cond_loss", mosfet_cond_loss, ...
                "mosfet_switch_loss", mosfet_switch_loss, ...
                "diode_cond_loss", diode_cond_loss, ...
                "L1_ESR_loss", L1_ESR_loss, ...
                "L2_ESR_loss", L2_ESR_loss, ...
                "C1_ESR_loss", C1_ESR_loss, ...
                "C2_ESR_loss", C2_ESR_loss, ...
                "efficiency", efficiency ...
            );

    out = struct(); 

    fields = fieldnames(L1_struct);
    for i = 1:numel(fields)
        out.(fields{i}) = L1_struct.(fields{i});
    end

    fields = fieldnames(L2_struct);
    for i = 1:numel(fields)
        out.(fields{i}) = L2_struct.(fields{i});
    end

    fields = fieldnames(C1_struct);
    for i = 1:numel(fields)
        out.(fields{i}) = C1_struct.(fields{i});
    end

    fields = fieldnames(Vout_struct);
    for i = 1:numel(fields)
        out.(fields{i}) = Vout_struct.(fields{i});
    end

    fields = fieldnames(loss_struct);
    for i = 1:numel(fields)
        out.(fields{i}) = loss_struct.(fields{i});
    end

    result = out;
catch
    error(result);
end_try_catch