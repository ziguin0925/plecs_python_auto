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

function out_struct = compute_leakage_current(target_time, raw_time_list, data, frequency, switch_on_time)

    name = 'leakage';
    out_struct = struct();

    time_idx = find(raw_time_list >= switch_on_time, 1);
    if isempty(time_idx)
        error("target_time 이후 데이터 없음");
    endif

    % After touch on time
    filtered_time = raw_time_list(time_idx:end);
    filtered_data = data(time_idx:end);
    operating_time = round(raw_time_list(end) * 1e4) / 1e4;
    step_per_period = round(length(raw_time_list) / (operating_time * frequency));

    % 특정 시간대의 값
    target_idx = find(filtered_time >= (switch_on_time + target_time), 1);
    out_struct.([name '_' num2str(target_time) 's']) = round(filtered_data(target_idx)* 1e4) / 1e4;

    % 10ms ~ 100ms rms 
    stamp_time = 0.02;
    for i = 1:1:5
        idx = find(filtered_time >= (switch_on_time + (i * stamp_time - 0.01) ), 1); % 20ms
        rms_val = sqrt(mean(filtered_data(idx:min(idx+step_per_period-1, length(filtered_data))).^2));
        out_struct.([name '_stamp_' num2str(i * stamp_time)]) = round(rms_val * 1e4) / 1e4;
    endfor

    % peak
    out_struct.([name '_peak']) = round(max(filtered_data) * 1e4) / 1e4;

    % min rms
    [min_val, min_idx]= min(filtered_data);
    out_struct.([name '_min_rms']) = round(sqrt(filtered_data(idx-step_per_period : idx).^2 / step_per_period)* 1e6) / 1e6

    threshold = 2e-3;

    sq = filtered_data.^2;
    S = cumsum(sq);
    idx_2mA = -1;
    disp(['step_per_period = ', num2str(step_per_period)]);
    for i = step_per_period * 2:step_per_period:length(sq)
        window_sum = S(i) - S(i-step_per_period);
        rms_val = sqrt(window_sum / step_per_period);

        if rms_val < threshold
            idx_2mA = i;
            break;
        end
    end
    if idx_2mA == -1
        out_struct.([name '_2mA_Time']) = Inf;
    else
        disp(['filtered_time(idx_2mA) : ' num2str(filtered_time(idx_2mA))]);
        out_struct.([name '_2mA_Time']) = round((filtered_time(idx_2mA) - switch_on_time) * 1e6) / 1e6;
    endif
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

    leakage_struct = compute_leakage_current(0.06, result.Time, result.Values(12, : ), 1000, touch_time);

    efficiency = max(result.Values(13, slice:before_touch_idx));

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

    fields = fieldnames(leakage_struct);
    for i = 1:numel(fields)
        out.(fields{i}) = leakage_struct.(fields{i});
    end

    result = out;
catch
    error(result);
end_try_catch