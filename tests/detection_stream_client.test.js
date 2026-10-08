const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../static/js/main.js'), 'utf8');
const streamSource = source.slice(source.indexOf('    function startImageDetection('), source.indexOf('    function updateProgressUI('));

async function runDetection(response) {
    const errors = [];
    const results = [];
    const context = {
        fetch: async () => response,
        TextDecoder,
        console,
        detectBtn: { disabled: false, textContent: '开始检测' },
        progressOverlay: { style: {} },
        resetProgressUI() {},
        updateProgressUI() {},
        showProgressError: message => errors.push(message),
        finishDetection: result => results.push(result),
    };
    vm.runInNewContext(streamSource, context);
    context.startImageDetection({});
    await new Promise(resolve => setImmediate(resolve));
    return { errors, results };
}

function streamedResponse(events) {
    const chunks = events.map(([type, data]) => Buffer.from(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`));
    return { ok: true, body: { getReader: () => ({ read: async () => chunks.length
        ? { done: false, value: chunks.shift() }
        : { done: true }
    }) } };
}

test('HTTP errors show the backend message and produce no detection result', async () => {
    const state = await runDetection({ ok: false, status: 500, text: async () => JSON.stringify({ msg: 'GPU 推理失败' }) });
    assert.deepEqual(state.errors, ['GPU 推理失败']);
    assert.equal(state.results.length, 0);
});

test('SSE errors produce no detection result', async () => {
    const state = await runDetection(streamedResponse([['error', { message: '检测链路失败' }]]));
    assert.deepEqual(state.errors, ['检测链路失败']);
    assert.equal(state.results.length, 0);
});

test('an HTML server error reports the HTTP status without rendering a result', async () => {
    const state = await runDetection({ ok: false, status: 500, text: async () => '<html>Internal Server Error</html>' });
    assert.match(state.errors[0], /HTTP 500/);
    assert.equal(state.results.length, 0);
});

test('a stream without a completion event reports an interrupted request', async () => {
    const state = await runDetection(streamedResponse([['progress', { step: 1 }]]));
    assert.match(state.errors[0], /未收到完整结果/);
    assert.equal(state.results.length, 0);
});

test('a complete stream renders the actual server result', async () => {
    const state = await runDetection(streamedResponse([['result', { label: 'real', confidence: 0.2 }], ['done', {}]]));
    assert.equal(state.errors.length, 0);
    assert.equal(state.results.length, 1);
    assert.equal(state.results[0].label, 'real');
    assert.equal(state.results[0].confidence, 0.2);
});
