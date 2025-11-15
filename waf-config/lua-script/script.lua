function main()
    local request_data = {
        method = m.getvar("REQUEST_METHOD") or "NO_METHOD",
        ip = m.getvar("REMOTE_ADDR") or "NO_IP"
    }
    local all_args = m.getvars("ARGS", {"none"})
    local params = {}
    for _, arg in ipairs(all_args) do
        params[arg["name"]] = arg["value"]
    end
    local payload = {
        request_data = request_data,
        parameters = params
    }
    local json = require("dkjson")
    local json_payload = json.encode(payload)
    local flask_url = "http://localhost:5000/predecir"
    local http = require("socket.http")
    local ltn12 = require("ltn12")
    local response_body = {}
    local res, code, response_headers = http.request({
        url = flask_url,
        method = 'POST',
        headers = {
            ["Content-Type"] = "application/json",
            ["Content-Length"] = #json_payload
        },
        source = ltn12.source.string(json_payload),
        sink = ltn12.sink.table(response_body)
    })
    m.log(1, "Enviando JSON a Flask: " .. json_payload)
    if code == 200 then
        m.log(1, "Datos enviados correctamente a Flask")
        local response_text = table.concat(response_body)
        m.log(1, "Respuesta de Flask: " .. response_text)
        if response_text == "Maliciosa" then
            m.log(1,"bloqueando solicitud...")
            m.setvar("tx.outbound_anomaly_score", "100")
            m.setvar("tx.anomaly_score", "100")
            m.setvar("tx.error_message","Acceso denegado por política de seguridad")
            m.setvar("tx.status", "403")
            return true
        else
            m.log(1, 'Permitiendo Solicitud')
            return false
        end
    else
        m.log(1, "Error al enviar datos a Flask. Codigo: " .. tostring(code))
    end
    return false
end