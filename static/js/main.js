// AI 伪造鉴别智能体 - 前端交互
document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('detect-form');
    if (!form) return;

    var type = form.getAttribute('data-type');
    var fileInput = document.getElementById('file-input');
    var uploadArea = document.getElementById('upload-area');
    var uploadError = document.getElementById('upload-error');
    var imageUrlInput = document.getElementById('image-url-input');
    var imageUrlImport = document.getElementById('image-url-import');
    var uploadPreviewUrl = null;
    var resultPreviewUrl = null;
    var resultStampObserver = null;
    var uploadValidationPending = false;
    var MIN_IMAGE_FILE_BYTES = 30 * 1024;
    var MAX_IMAGE_FILE_BYTES = 20 * 1024 * 1024;
    var MAX_BATCH_IMAGES = 200;
    var MIN_IMAGE_SHORT_EDGE = 300;
    var resultBackButton = document.getElementById('result-back-button');
    var contentSubtitle = document.querySelector('.content-subtitle');
    var detectBtn = document.getElementById('detect-btn');
    var retryBtn = document.getElementById('retry-btn');
    var processBtn = document.getElementById('process-btn');
    var processPanel = document.getElementById('process-panel');

    var uploadSection = document.getElementById('upload-section');
    var resultSection = document.getElementById('result-section');
    var previewContainer = document.getElementById('preview-container');
    var imagePreviewView = document.getElementById('image-preview-view');
    var aiCodePanel = document.getElementById('ai-code-panel');
    var aiCodeStatus = document.getElementById('ai-code-status');
    var inspectionTab = document.getElementById('image-tab-inspection');
    var aiCodeTab = document.getElementById('image-tab-ai-code');
    var aiCodeExport = document.getElementById('ai-code-export');

    // ── 进度条相关 DOM ──
    var progressOverlay = document.getElementById('progress-overlay');
    var progressTitleText = document.getElementById('progress-title-text');
    var progressSubText = document.getElementById('progress-sub-text');
    var progressBarFill = document.getElementById('progress-bar-fill');
    var progressStepIcon = document.getElementById('progress-step-icon');
    var progressErrorMsg = document.getElementById('progress-error-msg');
    var progressRetryBtn = document.getElementById('progress-retry-btn');
    var progressStepsRow = document.getElementById('progress-steps-row');
    var progressModal = progressOverlay.querySelector('.progress-modal');
    var progressPercent = document.getElementById('progress-percent');
    var progressBar = document.getElementById('progress-bar');
    var progressStatus = document.getElementById('progress-status');
    var progressStageCount = document.getElementById('progress-stage-count');
    var progressActivityList = document.getElementById('progress-activity-list');
    var progressElapsed = document.getElementById('progress-elapsed');
    var progressFootnote = document.getElementById('progress-footnote');
    var progressImage = document.getElementById('progress-image');
    var progressTimer = null;
    var progressStartedAt = 0;
    var progressValue = 0;
    var progressLastMessage = '';
    var progressPreviousFocus = null;

    // 存储最近一次检测详情，供「检测过程」面板使用
    var lastDetails = null;
    var lastNpr = null;
    var lastSpecialized = null;
    var lastCombined = null;
    var lastDeepfake = null;
    var lastDeepfakeCombined = null;
    var lastTamper = null;
    var lastTamperCombined = null;
    var lastFinalComparison = null;
    var lastShortcut = false;          // 水印短路判定（跳过深度学习）
    var lastWatermark = null;          // 水印检测结果（hidden_watermark_result / watermark_result）
    var lastHiddenWatermark = null;
    var lastWatermarkIsHidden = false; // 是否为隐式元数据标识（TC260/C2PA）
    var lastVisibleWatermark = null;   // 可见水印检测结果
    var lastContentAnalysis = null;    // 图片内容识别 + 知识库风险研判
    var aiIdentifierCheckComplete = false;
    var aiIdentifierLoading = false;

    // ========== AI 内容检测首页与图像检测页切换 ==========
    var landingPage = document.getElementById('landing-page');
    var topbar = document.getElementById('topbar');
    var appBody = document.getElementById('app-body');
    var homeNav = document.getElementById('nav-home');
    var landingEnterImage = document.getElementById('landing-enter-image');
    var landingHistoryBtn = document.getElementById('landing-history-btn');
    var landingTitle = document.getElementById('landing-card-title');
    var landingDescription = document.getElementById('landing-card-description');
    var landingNote = document.getElementById('landing-card-note');
    var landingVisual = document.getElementById('landing-upload-visual');

    var landingModeCopy = {
        image: { title: '图像 AI 伪造检测', description: '识别 AI 全图生成、深度伪造与图像篡改风险', note: '支持 JPG、PNG、BMP、WebP 格式 · 单张最大 20MB', action: '进入图像检测', icon: '▧' },
        video: { title: '视频内容检测', description: '识别视频中的深度伪造、合成与篡改风险', note: '视频检测能力正在接入，敬请期待', action: '视频检测即将上线', icon: '▷' },
        text: { title: '文本内容检测', description: '识别文本生成痕迹与潜在虚假信息风险', note: '文本检测能力正在接入，敬请期待', action: '文本检测即将上线', icon: 'T' },
        audio: { title: '音频内容检测', description: '识别语音克隆、合成音频与伪造风险', note: '音频检测能力正在接入，敬请期待', action: '音频检测即将上线', icon: '⌁' }
    };
    var activeLandingMode = 'image';

    function renderLandingMode(mode) {
        activeLandingMode = mode;
        var copy = landingModeCopy[mode] || landingModeCopy.image;
        document.querySelectorAll('.landing-mode').forEach(function (button) {
            button.classList.toggle('active', button.getAttribute('data-mode') === mode);
        });
        if (landingTitle) landingTitle.textContent = copy.title;
        if (landingDescription) landingDescription.textContent = copy.description;
        if (landingNote) landingNote.textContent = copy.note;
        if (landingEnterImage) landingEnterImage.textContent = copy.action;
        if (landingVisual) {
            var icon = landingVisual.querySelector('.landing-file-icon');
            if (icon) icon.textContent = copy.icon;
        }
    }

    function showLandingPage() {
        document.body.classList.remove('image-detection-page');
        if (landingPage) landingPage.style.display = '';
        if (topbar) topbar.style.display = 'none';
        if (appBody) appBody.style.display = 'none';
    }

    function showImageDetectionPage() {
        document.body.classList.add('image-detection-page');
        if (landingPage) landingPage.style.display = 'none';
        if (topbar) topbar.style.display = '';
        if (appBody) appBody.style.display = '';
        if (window.location.hash !== '#image') window.location.hash = 'image';
    }

    document.querySelectorAll('.landing-mode').forEach(function (button) {
        button.addEventListener('click', function () {
            renderLandingMode(button.getAttribute('data-mode'));
        });
    });
    if (landingEnterImage) {
        landingEnterImage.addEventListener('click', function () {
            if (activeLandingMode === 'image') showImageDetectionPage();
            else window.alert('该检测能力正在接入，当前可使用图像检测。');
        });
    }
    if (homeNav) {
        homeNav.addEventListener('click', function (event) {
            event.preventDefault();
            if (window.location.hash) window.location.hash = '';
            showLandingPage();
        });
    }
    var topbarHomeLink = document.getElementById('topbar-home-link');
    if (topbarHomeLink) {
        topbarHomeLink.addEventListener('click', function (event) {
            event.preventDefault();
            if (window.location.hash) window.location.hash = '';
            showLandingPage();
        });
    }
    var topbarBatchLink = document.getElementById('topbar-batch-link');
    if (topbarBatchLink) {
        topbarBatchLink.addEventListener('click', function () {
            var batchNav = document.getElementById('nav-batch-detect');
            if (batchNav) batchNav.click();
        });
    }
    if (landingHistoryBtn) {
        landingHistoryBtn.addEventListener('click', function () {
            showImageDetectionPage();
            var batchNav = document.getElementById('nav-batch-detect');
            if (batchNav) batchNav.click();
        });
    }
    window.addEventListener('hashchange', function () {
        if (window.location.hash === '#image') showImageDetectionPage();
        else showLandingPage();
    });
    renderLandingMode('image');
    if (window.location.hash === '#image') showImageDetectionPage();
    else showLandingPage();

    function updateButtonState() {
        var hasFile = fileInput && fileInput.files.length > 0;
        var countEl = document.getElementById('upload-selection-count');
        if (countEl) countEl.textContent = '已选择 ' + (hasFile ? fileInput.files.length : 0) + ' 张图片';
        if (hasFile && !uploadValidationPending) {
            detectBtn.classList.add('ready');
            detectBtn.disabled = false;
        } else {
            detectBtn.classList.remove('ready');
            detectBtn.disabled = true;
        }
    }

    function setUploadError(message) {
        if (!uploadError) return;
        uploadError.textContent = message || '';
        uploadError.hidden = !message;
    }

    function openBatchTaskFromUpload(files, source) {
        var imageFiles = Array.prototype.slice.call(files || []).filter(function (file) {
            return /\.(jpe?g|png|bmp|webp)$/i.test(file.name || '');
        });
        if (imageFiles.length === 0) {
            setUploadError('未选择支持的图片文件（JPG、JPEG、PNG、BMP 或 WebP）。');
            return;
        }
        if (imageFiles.length > MAX_BATCH_IMAGES) {
            setUploadError('单次最多上传 200 张图片，请分批选择。');
            return;
        }
        setUploadError('');
        clearUploadPreview();
        syncBatchOptionsFromSingle();
        showTaskCreateModal(imageFiles, source);
    }

    function refreshUploadSelection() {
        uploadValidationPending = false;
        setUploadError('');
        clearUploadPreview();
        var files = fileInput ? Array.prototype.slice.call(fileInput.files || []) : [];
        if (files.length > 1) {
            if (fileInput) fileInput.value = '';
            openBatchTaskFromUpload(files, 'files');
            updateButtonState();
            var selectedCount = document.getElementById('upload-selection-count');
            if (selectedCount && files.length <= MAX_BATCH_IMAGES) selectedCount.textContent = '已选择 ' + files.length + ' 张图片';
            return;
        }
        var file = fileInput.files[0];
        if (file) {
            var message = '';
            if (!/\.(jpe?g|png|bmp|webp)$/i.test(file.name)) message = '请选择 JPG、JPEG、PNG、BMP 或 WebP 图片。';
            else if (file.size === 0) message = '图片为空，请重新选择。';
            else if (file.size < MIN_IMAGE_FILE_BYTES) message = '图片文件大小不能小于 30 KB，请上传清晰原图。';
            else if (file.size > MAX_IMAGE_FILE_BYTES) message = '图片文件过大（超过 20 MB），请压缩后重新选择。';
            if (message) {
                fileInput.value = '';
                setUploadError(message);
            } else {
                uploadValidationPending = true;
                updateButtonState();
                var image = new Image();
                var imageUrl = URL.createObjectURL(file);
                image.onload = function () {
                    URL.revokeObjectURL(imageUrl);
                    if (!fileInput.files[0] || fileInput.files[0] !== file) return;
                    uploadValidationPending = false;
                    if (Math.min(image.naturalWidth, image.naturalHeight) < MIN_IMAGE_SHORT_EDGE) {
                        fileInput.value = '';
                        var sizeMessage = '上传图像最小尺寸不小于 300px（当前 ' + image.naturalWidth + ' × ' + image.naturalHeight + '）。';
                        setUploadError(sizeMessage);
                    } else {
                        showUploadPreview(file);
                    }
                    updateButtonState();
                };
                image.onerror = function () {
                    URL.revokeObjectURL(imageUrl);
                    if (!fileInput.files[0] || fileInput.files[0] !== file) return;
                    uploadValidationPending = false;
                    fileInput.value = '';
                    var unreadableMessage = '无法读取图片，请上传完整的 JPG、PNG、BMP 或 WebP 文件。';
                    setUploadError(unreadableMessage);
                    updateButtonState();
                };
                image.src = imageUrl;
            }
        }
        updateButtonState();
    }

    updateButtonState();
    if (fileInput) fileInput.addEventListener('change', refreshUploadSelection);
    if (imageUrlInput && imageUrlImport) {
        imageUrlInput.addEventListener('input', function () {
            imageUrlImport.disabled = !imageUrlInput.value.trim() || imageUrlImport.dataset.loading === 'true';
        });
        imageUrlImport.addEventListener('click', async function () {
            var urls = imageUrlInput.value.split(/[;,\n\r]+/).map(function (value) { return value.trim(); }).filter(Boolean);
            if (!urls.length) return;
            if (urls.length > MAX_BATCH_IMAGES) {
                setUploadError('单次最多导入 200 张图片，请分批输入。');
                return;
            }
            imageUrlImport.dataset.loading = 'true';
            imageUrlImport.disabled = true;
            imageUrlImport.textContent = '导入中';
            setUploadError('');
            try {
                var importedFiles = await Promise.all(urls.map(async function (url, index) {
                    var parsedUrl = new URL(url, window.location.href);
                    if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') throw new Error('图片地址仅支持 HTTP 或 HTTPS');
                    var response = await fetch(parsedUrl.href, { mode: 'cors' });
                    if (!response.ok) throw new Error('图片下载失败（HTTP ' + response.status + '）');
                    var blob = await response.blob();
                    var filename = decodeURIComponent(parsedUrl.pathname.split('/').pop() || ('image-' + (index + 1) + '.jpg')).split('?')[0];
                    if (!/\.(jpe?g|png|bmp|webp)$/i.test(filename)) {
                        var extension = (blob.type.split('/')[1] || '').replace('jpeg', 'jpg').replace('svg+xml', '');
                        if (!/^(jpg|png|bmp|webp)$/i.test(extension)) throw new Error('仅支持 JPG、PNG、BMP、WebP 图片地址');
                        filename += '.' + extension;
                    }
                    if (blob.size < MIN_IMAGE_FILE_BYTES || blob.size > MAX_IMAGE_FILE_BYTES) throw new Error('图片大小需在 30 KB–20 MB 之间');
                    return new File([blob], filename, { type: blob.type || 'image/jpeg' });
                }));
                if (importedFiles.length > 1) {
                    openBatchTaskFromUpload(importedFiles, 'url');
                    var countEl = document.getElementById('upload-selection-count');
                    if (countEl) countEl.textContent = '已选择 ' + importedFiles.length + ' 张图片';
                } else {
                    var transfer = new DataTransfer();
                    transfer.items.add(importedFiles[0]);
                    fileInput.files = transfer.files;
                    refreshUploadSelection();
                }
                imageUrlInput.value = '';
            } catch (error) {
                setUploadError(error.name === 'TypeError' ? '图片地址无法读取，请确认地址可访问，且图片服务器允许跨域访问。' : (error.message || '导入失败，请确认图片地址可公开访问且允许跨域读取'));
            } finally {
                imageUrlImport.dataset.loading = 'false';
                imageUrlImport.textContent = '导入';
                imageUrlImport.disabled = !imageUrlInput.value.trim();
            }
        });
    }
    document.querySelectorAll('[data-open-batch]').forEach(function (button) {
        button.addEventListener('click', function () {
            var batchNav = document.getElementById('nav-batch-detect');
            if (batchNav) batchNav.click();
        });
    });

    // 拖拽上传
    if (uploadArea) {
        ['dragenter', 'dragover'].forEach(function (eventName) {
            uploadArea.addEventListener(eventName, function (e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(function (eventName) {
            uploadArea.addEventListener(eventName, function (e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });
        });
        uploadArea.addEventListener('drop', function (e) {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0 && fileInput) {
                if (e.dataTransfer.files.length > 1) {
                    openBatchTaskFromUpload(e.dataTransfer.files, 'files');
                    return;
                }
                fileInput.files = e.dataTransfer.files;
                refreshUploadSelection();
            }
        });
    }

    function returnToUploadPage() {
        if (uploadSection) uploadSection.style.display = '';
        if (resultSection) resultSection.style.display = 'none';
        if (detectBtn) detectBtn.style.display = '';
        if (retryBtn) retryBtn.style.display = 'none';
        if (processBtn) processBtn.style.display = 'none';
        if (processPanel) processPanel.style.display = 'none';
        if (resultBackButton) resultBackButton.style.display = 'none';
        if (contentSubtitle) contentSubtitle.style.display = '';
        document.body.classList.remove('result-page-active');
        form.classList.remove('result-workspace');
        lastDetails = null;
        lastShortcut = false;
        lastWatermark = null;
        lastHiddenWatermark = null;
        lastWatermarkIsHidden = false;
        lastVisibleWatermark = null;
        lastContentAnalysis = null;
        aiIdentifierCheckComplete = false;
        aiIdentifierLoading = false;
        setImageResultView('inspection');
        detectBtn.textContent = '开始检测';
        setUploadError('');
        if (fileInput) fileInput.value = '';
        clearUploadPreview();
        updateButtonState();
    }

    if (retryBtn) retryBtn.addEventListener('click', returnToUploadPage);
    if (resultBackButton) resultBackButton.addEventListener('click', returnToUploadPage);

    // 检测过程按钮
    if (processBtn && processPanel) {
        processBtn.addEventListener('click', function () {
            var panel = processPanel;
            if (panel.style.display === 'none' || panel.style.display === '') {
                panel.style.display = '';
                populateProcessPanel();
                processBtn.classList.add('active');
            } else {
                panel.style.display = 'none';
                processBtn.classList.remove('active');
            }
        });
    }

    // 三类专项检测明细默认收起，标题或小三角均可展开/隐藏。
    function setDetailGroupCollapsed(group, collapsed) {
        if (!group) return;
        var button = group.querySelector('.detection-detail-toggle');
        var content = group.querySelector('.detection-detail-content');
        if (!button || !content) return;
        button.setAttribute('aria-expanded', String(!collapsed));
        content.hidden = collapsed;
        var hint = button.querySelector('.detail-toggle-hint');
        if (hint) hint.textContent = group.classList.contains('scoring-detail-group') ? (collapsed ? '详情' : '收起') : (collapsed ? '点击查看模型明细' : '点击收起模型明细');
    }

    function resetDetailGroups() {
        document.querySelectorAll('.detection-detail-group').forEach(function (group) {
            setDetailGroupCollapsed(group, true);
        });
    }

    document.querySelectorAll('.detection-detail-toggle').forEach(function (button) {
        button.addEventListener('click', function () {
            var group = button.closest('.detection-detail-group');
            setDetailGroupCollapsed(group, button.getAttribute('aria-expanded') === 'true');
        });
    });

    function getSelectedModels(container) {
        var root = container || document;
        return Array.prototype.slice.call(root.querySelectorAll('.upload-model-input:checked'))
            .map(function (input) { return input.value; });
    }

    function getSelectedCrimeScene(container) {
        var root = container || document;
        var selected = root.querySelector('.crime-scene-input:checked');
        return selected ? selected.value : '';
    }

    function syncModelCards(container) {
        var root = container || document;
        var inputs = root.querySelectorAll('.upload-model-input');
        inputs.forEach(function (input) {
            var card = input.closest('.upload-model-card');
            if (card) card.classList.toggle('is-selected', input.checked);
        });
        var status = document.getElementById('upload-models-status');
        if (status && (!container || container.closest('#detect-form'))) {
            var count = getSelectedModels(root).length;
            status.textContent = count ? ('已选 ' + count + ' 项' + (count === 3 ? '联合检测' : '专项检测')) : '请至少选择 1 项检测';
        }
        var selectAll = document.getElementById('select-all-models');
        if (selectAll && root.closest && root.closest('#detect-form')) {
            selectAll.checked = inputs.length > 0 && Array.prototype.every.call(inputs, function (input) { return input.checked; });
            selectAll.indeterminate = !selectAll.checked && Array.prototype.some.call(inputs, function (input) { return input.checked; });
        }
    }

    function syncCrimeSceneCards(container) {
        var root = container || document;
        var inputs = root.querySelectorAll('.crime-scene-input');
        inputs.forEach(function (input) {
            var card = input.closest('.crime-scene-card');
            if (card) card.classList.toggle('is-selected', input.checked);
        });
        var status = document.getElementById('crime-scenes-status');
        if (status && (!container || container.closest('#detect-form'))) {
            var labels = { fraud: '涉诈图片', pornography: '涉黄图片', rumor: '涉谣言图片' };
            var scene = getSelectedCrimeScene(root);
            status.textContent = scene ? ('已选：' + labels[scene]) : '未选择：仅鉴别 AI 生成';
        }
    }

    document.querySelectorAll('.upload-model-input').forEach(function (input) {
        input.addEventListener('change', function () { syncModelCards(input.closest('.upload-model-grid')); });
    });
    var selectAllModels = document.getElementById('select-all-models');
    if (selectAllModels) {
        selectAllModels.addEventListener('change', function () {
            document.querySelectorAll('#single-models .upload-model-input').forEach(function (input) { input.checked = selectAllModels.checked; });
            syncModelCards(document.getElementById('single-models'));
        });
    }
    document.querySelectorAll('.crime-scene-input').forEach(function (input) {
        input.addEventListener('change', function () {
            var grid = input.closest('.crime-scene-grid');
            if (input.checked && grid) {
                grid.querySelectorAll('.crime-scene-input').forEach(function (otherInput) {
                    if (otherInput !== input) otherInput.checked = false;
                });
            }
            syncCrimeSceneCards(grid);
            updateBatchStartBtn();
        });
    });
    syncModelCards(document.querySelector('#detect-form .upload-model-grid'));
    syncModelCards(document.getElementById('batch-models'));
    syncCrimeSceneCards(document.querySelector('#detect-form .crime-scene-grid'));
    syncCrimeSceneCards(document.getElementById('batch-crime-scenes'));

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        if (detectBtn.disabled) return;

        if (getSelectedModels(form).length === 0) {
            alert('请至少选择一项检测模型');
            return;
        }
        var formData = new FormData(form);

        // 使用 SSE 流式图像检测
        startImageDetection(formData);
    });

    function showResult(data) {
        // 存储详情，供检测过程面板使用
        lastDetails = data.details || null;
        lastNpr = data.npr || null;
        lastSpecialized = data.specialized_models || null;
        lastCombined = data.combined || null;
        lastDeepfake = data.deepfake || null;
        lastDeepfakeCombined = data.deepfake_combined || null;
        lastTamper = data.tamper || null;
        lastTamperCombined = data.tamper_combined || null;
        lastFinalComparison = data.final_comparison || null;
        // 内容识别只在后端已确认 AI 生成时才允许展示；同时兼容旧任务数据。
        lastContentAnalysis = data.label === 'ai_generated' ? (data.content_analysis || null) : null;

        // 水印短路判定：标记并保存水印检测结果（兼容流式/非流式两种返回）
        var hiddenResult = data.hidden_watermark_result || {};
        lastHiddenWatermark = data.hidden_watermark_result || null;
        var c2paPresent = !!(
            (hiddenResult.c2pa_verification && hiddenResult.c2pa_verification.present) ||
            (hiddenResult.metadata && hiddenResult.metadata.c2pa && hiddenResult.metadata.c2pa.present)
        );
        lastShortcut = !c2paPresent && (!!data.shortcut ||
            !!hiddenResult.detected || !!(data.watermark_result && data.watermark_result.detected));
        lastWatermarkIsHidden = !!hiddenResult.detected;
        lastWatermark = hiddenResult.detected ? hiddenResult : (data.watermark_result && data.watermark_result.detected ? data.watermark_result : data.hidden_watermark_result || data.watermark_result || null);
        lastVisibleWatermark = data.watermark_result || null;
        aiIdentifierCheckComplete = !data.demo_mode &&
            data.hidden_watermark_result !== undefined && data.watermark_result !== undefined;

        // 判定结果直接叠加在图片上。
        var labelCode = data.label || 'uncertain';

        // 1. 切换显示
        if (uploadSection) uploadSection.style.display = 'none';
        if (resultSection) resultSection.style.display = '';
        if (detectBtn) detectBtn.style.display = 'none';
        if (retryBtn) retryBtn.style.display = '';
        if (resultBackButton) resultBackButton.style.display = 'inline-flex';
        if (contentSubtitle) contentSubtitle.style.display = 'none';
        document.body.classList.add('result-page-active');
        form.classList.add('result-workspace');
        if (processPanel) processPanel.style.display = lastShortcut ? 'none' : '';
        ['combined-section', 'npr-section', 'neural-analysis-section', 'df-combined-section', 'df-ela-section', 'df-model-card', 'tp-combined-section', 'tp-forensic-section', 'tp-pf-section', 'watermark-notice', 'content-risk-section'].forEach(function (id) {
            var section = document.getElementById(id);
            if (section) section.style.display = 'none';
        });
        resetDetailGroups();

        // 2. 填充预览内容
        previewContainer.innerHTML = buildPreview(data);
        renderImageVerdictStamp(labelCode);
        populateAICodePanel();
        setImageResultView('inspection');

        // 展示融合判定结果
        if (lastCombined) {
            populateCombinedVerdict(lastCombined);
        }

        // 展示 DeepFake 检测结果（无人脸时跳过子模块，仅显示综合判定）
        var noFace = lastDeepfakeCombined && !lastDeepfakeCombined.face_detected;
        if (lastDeepfake && !noFace) {
            populateDeepfakeResults(lastDeepfake);
        } else {
            hideDeepfakeSections();
        }

        // 展示 DeepFake 综合判定 + 纯模型
        if (lastDeepfakeCombined) {
            populateDFCombinedVerdict(lastDeepfakeCombined);
        }
        if (lastSpecialized && lastSpecialized.deepfake_detector && !noFace) {
            populateDFModelCard(lastSpecialized.deepfake_detector);
        } else {
            var dfmc = document.getElementById('df-model-card');
            if (dfmc) dfmc.style.display = 'none';
        }

        // ── 图像篡改 专项检测结果 ──
        if (lastTamper) {
            populateTamperCombinedVerdict(lastTamperCombined);
            populateTamperResults(lastTamper);
        } else {
            hideTamperSections();
        }

        // 结果工作台会在右栏即时呈现检测信息，无需再单独展开检测过程。
        if (lastDetails) {
            populateProcessPanel();
            if (processBtn) {
                processBtn.style.display = 'none';
                processBtn.classList.remove('active');
            }
        } else if (processBtn) {
            processBtn.style.display = 'none';
        }
        renderScoringGroups();
    }

    function renderScoringGroups() {
        var groups = [
            { key: 'ai', title: 'AI 全图生成', result: lastCombined, raw: lastNpr, badge: '物理分析 + 神经网络' },
            { key: 'df', title: '深度伪造', result: lastDeepfakeCombined, raw: lastDeepfake, badge: 'ELA + ViT-B' },
            { key: 'tp', title: '图像篡改', result: lastTamperCombined, raw: lastTamper, badge: 'TruFor 主模型' }
        ];
        function valid(value) { return typeof value === 'number' && isFinite(value); }
        function clamp(value) { return valid(value) ? Math.max(0, Math.min(1, value)) : 0; }
        function points(value) { return valid(value) ? (value * 100).toFixed(2) + ' 分' : '未提供'; }
        function percent(value) { return valid(value) ? (value * 100).toFixed(2) + '%' : '未提供'; }
        function numeric(value, digits) { return valid(value) ? value.toFixed(digits) : '未提供'; }
        function tone(value) { return !valid(value) ? 'muted' : value >= 0.6 ? 'danger' : 'safe'; }
        function metric(label, value, display, color) {
            return '<div class="analysis-metric"><span>' + escapeHtml(label) + '</span><b>' + escapeHtml(display) + '</b><div class="analysis-metric-track" aria-hidden="true"><i class="' + (valid(value) ? (color || 'neutral') : 'muted') + '" style="width:' + (clamp(value) * 100).toFixed(2) + '%"></i></div></div>';
        }
        function ring(value, label) {
            return '<div class="analysis-gauge ' + tone(value) + '"><div class="analysis-gauge-ring" role="img" aria-label="' + escapeHtml(label + '：' + points(value)) + '" style="--gauge-value:' + (clamp(value) * 100).toFixed(2) + '%"><strong>' + (valid(value) ? (value * 100).toFixed(1) : '--') + '</strong></div><small>' + escapeHtml(label) + '</small></div>';
        }
        function card(title, badge, value, metrics, extra, showRing) {
            return '<article class="analysis-model-card"><header><h4>' + escapeHtml(title) + '</h4><span class="analysis-model-badge">' + escapeHtml(badge) + '</span></header><div class="analysis-model-body">' + (showRing ? ring(value, '分析得分') : '') + '<div class="analysis-metrics">' + metrics + '</div></div>' + (extra || '') + '</article>';
        }
        function status(text) { return '<p class="analysis-model-status">' + escapeHtml(text) + '</p>'; }
        function findComponent(components, name) { return components.find(function (item) { return item.name === name; }); }
        function modelCards(group, components) {
            var html = '', raw = group.raw || {};
            if (group.key === 'ai') {
                if (group.raw) {
                    var features = raw.features || {};
                    var featureDefs = [
                        ['峰值密度', 'peak_ratio', function (v) { return v * 100; }],
                        ['频谱平坦度', 'spectral_flatness', function (v) { return (1 - v) * 0.8; }],
                        ['高频能量比', 'high_freq_ratio', function (v) { return v * 1.5; }],
                        ['径向方差', 'radial_variance', function (v) { return v * 50; }]
                    ];
                    var metrics = featureDefs.map(function (item) {
                        var value = features[item[1]];
                        return metric(item[0], valid(value) ? item[2](value) : null, numeric(value, 6), 'purple');
                    }).join('');
                    html += card('NPR 噪声模式分析', '物理分析', raw.verdict === 'error' ? null : raw.score, metrics, raw.verdict === 'error' ? status('分析失败') : '', true);
                    if (raw.physical_fused) {
                        var dimensions = raw.physical_dimensions || {};
                        var physicalMetrics = [['采集痕迹弱度', 'camera_trace_weakness'], ['噪声弱度', 'noise_weakness'], ['CFA 弱度', 'cfa_weakness'], ['元数据弱度', 'metadata_weakness']].map(function (item) {
                            return metric(item[0], dimensions[item[1]], percent(dimensions[item[1]]), 'purple');
                        }).join('');
                        html += card('相机成像物证', '物理分析', raw.physical_score, physicalMetrics, '', true);
                    }
                }
                [['univfd', 'UnivFD'], ['probe_dinov2', 'PROBE-DINOv2']].forEach(function (item) {
                    var model = (lastSpecialized || {})[item[0]];
                    var component = findComponent(components, item[1]);
                    if (!model && !component) return;
                    model = model || {};
                    var available = model.available !== false && (valid(model.ai_score) || !!component);
                    var aiScore = available ? (valid(model.ai_score) ? model.ai_score : component.score) : null;
                    var modelMetrics = metric('AI 生成分', aiScore, percent(aiScore), 'danger') + metric('真实图片分', available ? model.human_score : null, available ? percent(model.human_score) : '未提供', 'safe');
                    var extra = available ? '' : status('模型不可用');
                    html += card(model.label || item[1], '神经网络', aiScore, modelMetrics, extra, false);
                });
            } else if (group.key === 'df') {
                var ela = raw.ela;
                if (ela) {
                    var metrics = metric('异常像素比', ela.anomaly_ratio, percent(ela.anomaly_ratio), 'danger') + metric('聚集区域比', ela.cluster_ratio, percent(ela.cluster_ratio), 'danger') + metric('平均误差', valid(ela.mean_error) ? ela.mean_error / 15 : null, numeric(ela.mean_error, 2), 'neutral') + metric('误差标准差', valid(ela.std_error) ? ela.std_error / 30 : null, numeric(ela.std_error, 2), 'neutral');
                    html += card('ELA 压缩误差分析', '辅助分析', ela.verdict === 'error' ? null : ela.score, metrics, ela.verdict === 'error' ? status('分析失败') : group.result.face_detected === false ? status('无人脸，仅供辅助参考') : '', true);
                }
                var model = (lastSpecialized || {}).deepfake_detector || {};
                var component = findComponent(components, '深度伪造人脸模型');
                var available = model.available !== false && model.verdict !== '检测失败' && (valid(model.deepfake_score) || !!(component && component.available !== false));
                var fake = available ? (valid(model.deepfake_score) ? model.deepfake_score : component.score) : null;
                var modelMetrics = metric('深度伪造分', fake, percent(fake), 'danger') + metric('真实图片分', available ? model.real_score : null, available ? percent(model.real_score) : '未提供', 'safe');
                html += card(model.label || 'DeepFake Detector v2', 'ViT-B', fake, modelMetrics, !available ? status('模型未提供有效评分') : group.result.face_detected === false ? status('无人脸') : '', false);
            } else {
                var trufor = raw.trufor || {};
                var available = trufor.available === true;
                var metrics = metric('篡改评分', available ? trufor.score : null, available ? points(trufor.score) : '未提供', 'danger') + metric('模型可靠性', available ? trufor.reliability : null, available ? percent(trufor.reliability) : '未提供', 'safe') + metric('异常区域占比', available ? trufor.map_coverage : null, available ? percent(trufor.map_coverage) : '未提供', 'purple');
                var maps = [];
                [['map_png', '篡改定位图'], ['confidence_png', '可靠性图']].forEach(function (item) {
                    if (!available || !trufor[item[0]]) return;
                    var image = document.createElement('img');
                    image.src = trufor[item[0]];
                    image.alt = 'TruFor ' + item[1];
                    image.className = 'analysis-localization-image';
                    maps.push('<figure>' + image.outerHTML + '<figcaption>' + item[1] + '</figcaption></figure>');
                });
                var extra = available ? '' : status('模型不可用');
                if (maps.length) extra += '<div class="analysis-localization-grid">' + maps.join('') + '</div>';
                html += card('TruFor 篡改定位', '主模型', available ? trufor.score : null, metrics, extra, true);
                [['copy_move', '复制移动分析'], ['block_noise', '区块噪声分析']].forEach(function (item) {
                    var evidence = raw[item[0]];
                    if (!evidence) return;
                    var metrics = metric('辅助评分', evidence.score, points(evidence.score), 'purple');
                    if (item[0] === 'copy_move') metrics += metric('匹配点数量', valid(evidence.match_count) ? evidence.match_count / 200 : null, numeric(evidence.match_count, 0), 'neutral');
                    if (item[0] === 'block_noise') metrics += metric('噪声离散度', valid(evidence.dispersion) ? evidence.dispersion * 5 : null, numeric(evidence.dispersion, 4), 'neutral');
                    html += card(item[1], '辅助证据', evidence.score, metrics, '', false);
                });
            }
            return html;
        }
        groups.forEach(function (group) {
            var container = document.getElementById(group.key + '-detail-group');
            var score = document.getElementById(group.key + '-detail-score');
            var detail = document.getElementById(group.key + '-scoring-process');
            if (!container || !detail || !score) return;
            container.style.display = '';
            var result = group.result;
            if (lastShortcut || !result) {
                score.textContent = lastShortcut ? '已跳过' : '未检测';
                detail.innerHTML = '<div class="analysis-empty-state">' + (lastShortcut ? '已根据水印或平台标识完成判定，本次未执行该项模型评分。具体依据见水印核验结果。' : '本次未执行该项检测，没有可展示的评分过程。') + '</div>';
                return;
            }
            var scoring = result.scoring || {};
            var components = scoring.components || [];
            var unavailable = group.key === 'tp' && result.model_score == null;
            var noFace = group.key === 'df' && result.face_detected === false;
            var state = unavailable || noFace ? 'muted' : valid(scoring.threshold) ? ((scoring.operator === '>' ? result.final_score > scoring.threshold : result.final_score >= scoring.threshold) ? 'danger' : 'safe') : tone(result.final_score);
            var scoreLabel = unavailable ? '不可用' : noFace ? '无人脸' : points(result.final_score);
            score.textContent = scoreLabel;
            var html = '<div class="analysis-category-divider"><span>' + group.title + '专项检测</span></div>';
            html += '<section class="analysis-verdict-card ' + state + '"><header><h4>' + group.title + '综合判定</h4><span class="analysis-verdict-badge">' + group.badge + '</span></header><div class="analysis-verdict-main"><span>综合得分</span><strong>' + (unavailable || noFace ? '--' : valid(result.final_score) ? (result.final_score * 100).toFixed(2) : '--') + '</strong><small>分</small><b>' + escapeHtml(result.final_verdict || '等待判定') + '</b></div>';
            html += '</section>' + modelCards(group, components);
            detail.innerHTML = html;
        });
    }

    function buildPreview(data) {
        if (resultPreviewUrl) URL.revokeObjectURL(resultPreviewUrl);
        resultPreviewUrl = null;
        // 已上传文件 → 图片预览
        if (data.filename && fileInput && fileInput.files.length > 0) {
            var file = fileInput.files[0];
            var ext = file.name.split('.').pop().toLowerCase();

            // 图片预览
            if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].indexOf(ext) !== -1) {
                var img = document.createElement('img');
                resultPreviewUrl = URL.createObjectURL(file);
                img.src = resultPreviewUrl;
                img.alt = file.name;
                return img.outerHTML;
            }

            // 未知文件类型（理论上不会出现）
            return '<div class="preview-generic">' +
                '<p class="preview-filename">' + file.name + '</p>' +
                '<p class="preview-type">文件</p>' +
                '</div>';
        }

        return '<p class="preview-empty">暂无图片预览，请重新上传图片</p>';
    }

    function renderImageVerdictStamp(labelCode) {
        if (!previewContainer) return;
        if (resultStampObserver) resultStampObserver.disconnect();
        resultStampObserver = null;
        var oldLabel = previewContainer.querySelector('.image-verdict-stamp');
        if (oldLabel) oldLabel.remove();
        var image = previewContainer.querySelector('img');
        if (!image) return;
        var verdicts = {
            real: { text: '真实图片', style: 'real' },
            ai_generated: { text: '含AI生成合成', style: 'forged' },
            suspected_ai: { text: '疑似AI合成', style: 'suspected' }
        };
        var verdict = verdicts[labelCode] || { text: '待核验', style: 'uncertain' };
        var label = document.createElement('div');
        label.className = 'image-verdict-stamp stamp-' + verdict.style;
        label.setAttribute('role', 'status');
        label.setAttribute('aria-label', '检测结论：' + verdict.text);
        label.innerHTML = '<svg class="verdict-stamp-ring" viewBox="0 0 160 160" aria-hidden="true"><circle cx="80" cy="80" r="67"/><circle class="verdict-stamp-inner" cx="80" cy="80" r="51"/><g class="verdict-stamp-stars"><text x="80" y="35">★</text><text x="43" y="49">★</text><text x="117" y="49">★</text><text x="80" y="139">★</text><text x="43" y="125">★</text><text x="117" y="125">★</text></g></svg>';
        var ribbon = document.createElement('strong');
        ribbon.className = 'verdict-stamp-ribbon';
        ribbon.textContent = verdict.text;
        label.appendChild(ribbon);
        previewContainer.appendChild(label);

        function positionStamp() {
            if (!image.naturalWidth || !image.naturalHeight) return;
            // object-fit 留出的空白不属于图片，把印章贴在实际图片边界内。
            var scale = Math.min(image.clientWidth / image.naturalWidth, image.clientHeight / image.naturalHeight);
            var insetX = (image.clientWidth - image.naturalWidth * scale) / 2;
            var insetY = (image.clientHeight - image.naturalHeight * scale) / 2;
            label.style.top = (image.offsetTop + insetY + 16) + 'px';
            label.style.right = (previewContainer.clientWidth - image.offsetLeft - image.clientWidth + insetX + 16) + 'px';
        }
        image.addEventListener('load', positionStamp, { once: true });
        if (typeof ResizeObserver !== 'undefined') {
            resultStampObserver = new ResizeObserver(positionStamp);
            resultStampObserver.observe(previewContainer);
        }
        positionStamp();
    }

    // ==================== AI 标识编码核验 ====================
    function setImageResultView(view) {
        var isCode = view === 'code';
        if (imagePreviewView) imagePreviewView.hidden = isCode;
        if (aiCodePanel) {
            aiCodePanel.hidden = !isCode;
            aiCodePanel.style.display = isCode ? '' : 'none';
        }
        if (inspectionTab) {
            inspectionTab.classList.toggle('active', !isCode);
            inspectionTab.setAttribute('aria-selected', String(!isCode));
        }
        if (aiCodeTab) {
            aiCodeTab.classList.toggle('active', isCode);
            aiCodeTab.setAttribute('aria-selected', String(isCode));
        }
    }

    function populateAICodePanel() {
        var hidden = lastHiddenWatermark || {};
        var tc260 = hidden.tc260 || {};
        var fields = tc260.fields || {};
        var visible = lastVisibleWatermark || {};
        var metadata = hidden.metadata || {};
        var hasFields = Object.keys(fields).length > 0;

        function setCodeField(id, value, fallback) {
            var el = document.getElementById(id);
            if (!el) return;
            var hasValue = value !== undefined && value !== null && String(value).trim() !== '';
            el.textContent = hasValue ? String(value) : (fallback || '未检出');
            el.classList.toggle('found', hasValue);
        }

        var implicitStatus = hasFields ? '已检出（TC260:AIGC）' : null;
        if (!implicitStatus && metadata.detected) {
            implicitStatus = '已检出（' + (metadata.source || '元数据标识') + '）';
        }
        setCodeField('ai-code-aigc-field', implicitStatus);
        var label = fields.Label;
        setCodeField('ai-code-label', label === '1' ? 'AI生成（1）' : label);
        setCodeField('ai-code-producer', fields.ContentProducer);
        setCodeField('ai-code-produce-id', fields.ProduceID);
        setCodeField('ai-code-reserved-1', fields.ReservedCode1);
        setCodeField('ai-code-propagator', fields.ContentPropagator);
        setCodeField('ai-code-propagate-id', fields.PropagateID);
        setCodeField('ai-code-reserved-2', fields.ReservedCode2);
        setCodeField('ai-code-text-watermark', visible.detected ? (visible.detected_text || '已检出可见标识') : null);
        setCodeField('ai-code-logo-watermark', visible.detected && visible.source ? ('平台标识：' + visible.source) : null);
        if (aiIdentifierCheckComplete && !aiIdentifierLoading) {
            setAiCodeStatus('已完成图像隐式标识与可见水印核验。', 'success');
        }
    }

    function setAiCodeStatus(message, state) {
        if (!aiCodeStatus) return;
        aiCodeStatus.textContent = message || '';
        aiCodeStatus.className = 'ai-code-status' + (state ? ' ' + state : '');
    }

    function loadAiIdentifierResults() {
        if (aiIdentifierCheckComplete || aiIdentifierLoading || !fileInput || !fileInput.files.length) return;
        aiIdentifierLoading = true;
        setAiCodeStatus('正在核验图像隐式标识与可见水印…', 'loading');

        var requestData = new FormData();
        requestData.append('file', fileInput.files[0]);
        fetch('/api/detect/image/identifiers', { method: 'POST', body: requestData })
            .then(function (response) {
                if (!response.ok) throw new Error('核验服务响应异常（' + response.status + '）');
                return response.json();
            })
            .then(function (payload) {
                if (!payload || payload.code !== 200 || !payload.data) {
                    throw new Error((payload && payload.msg) || '标识核验未返回结果');
                }
                lastWatermark = payload.data.hidden_watermark_result || null;
                lastHiddenWatermark = payload.data.hidden_watermark_result || null;
                lastWatermarkIsHidden = !!payload.data.hidden_watermark_result;
                lastVisibleWatermark = payload.data.watermark_result || null;
                aiIdentifierCheckComplete = true;
                aiIdentifierLoading = false;
                populateAICodePanel();
            })
            .catch(function (error) {
                setAiCodeStatus('核验暂不可用：' + error.message, 'error');
            })
            .finally(function () {
                aiIdentifierLoading = false;
            });
    }

    function exportImageHex() {
        if (!fileInput || !fileInput.files || !fileInput.files.length) return;
        var file = fileInput.files[0];
        var reader = new FileReader();
        reader.onload = function () {
            var bytes = new Uint8Array(reader.result);
            var lines = ['# AI 标识编码核验导出', '# 文件：' + file.name, '# 格式：十六进制（每行 16 字节）', ''];
            for (var offset = 0; offset < bytes.length; offset += 16) {
                var row = [];
                for (var i = offset; i < Math.min(offset + 16, bytes.length); i++) {
                    row.push(bytes[i].toString(16).padStart(2, '0').toUpperCase());
                }
                lines.push(offset.toString(16).padStart(8, '0').toUpperCase() + '  ' + row.join(' '));
            }
            var blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
            var url = URL.createObjectURL(blob);
            var link = document.createElement('a');
            link.href = url;
            link.download = file.name + '.ai-identifier.hex.txt';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        };
        reader.readAsArrayBuffer(file);
    }

    if (inspectionTab) inspectionTab.addEventListener('click', function () { setImageResultView('inspection'); });
    if (aiCodeTab) aiCodeTab.addEventListener('click', function () {
        setImageResultView('code');
        loadAiIdentifierResults();
    });
    if (aiCodeExport) aiCodeExport.addEventListener('click', exportImageHex);

    

    // 判断是否为图片文件
    function isImageFile(file) {
        var ext = file.name.split('.').pop().toLowerCase();
        return ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].indexOf(ext) !== -1;
    }

    // 在 upload-area 中立即显示图片预览
    function showUploadPreview(file) {
        if (!uploadArea) return;
        uploadArea.classList.add('has-preview');
        if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
        // 移除已有预览
        var old = uploadArea.querySelector('.upload-preview-img');
        if (old) old.remove();
        // 创建预览图
        var img = document.createElement('img');
        img.className = 'upload-preview-img';
        uploadPreviewUrl = URL.createObjectURL(file);
        img.src = uploadPreviewUrl;
        img.alt = file.name;
        uploadArea.appendChild(img);
    }

    // 清除 upload-area 中的图片预览，恢复初始状态
    function clearUploadPreview() {
        if (!uploadArea) return;
        if (resultStampObserver) resultStampObserver.disconnect();
        resultStampObserver = null;
        if (resultPreviewUrl) URL.revokeObjectURL(resultPreviewUrl);
        resultPreviewUrl = null;
        uploadArea.classList.remove('has-preview');
        if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
        uploadPreviewUrl = null;
        var old = uploadArea.querySelector('.upload-preview-img');
        if (old) old.remove();
    }

    // ==================== 雷达图 ====================
    function drawRadarChart(aiScore, dfScore, spScore) {
        var svg = document.getElementById('radar-chart');
        if (!svg) return;
        var cx = 110, cy = 105, r = 58;

        var angles = [
            { x: cx, y: cy - r },                                    // 顶部
            { x: cx - r * 0.866, y: cy + r * 0.5 },                 // 左下
            { x: cx + r * 0.866, y: cy + r * 0.5 }                  // 右下
        ];

        var html = '';
        // 4 层同心三角网格
        for (var lv = 1; lv <= 4; lv++) {
            var scale = lv / 4;
            var pts = [];
            for (var a = 0; a < 3; a++) {
                pts.push((cx + (angles[a].x - cx) * scale) + ',' + (cy + (angles[a].y - cy) * scale));
            }
            html += '<polygon points="' + pts.join(' ') + '" fill="none" stroke="#e5e7eb" stroke-width="1"/>';
        }
        // 轴线
        for (var a = 0; a < 3; a++) {
            html += '<line x1="' + cx + '" y1="' + cy + '" x2="' + angles[a].x + '" y2="' + angles[a].y + '" stroke="#eee" stroke-width="1"/>';
        }
        // 刻度标签
        var labY = cy - r * 0.78;
        html += '<text x="' + cx + '" y="' + labY + '" font-size="8" fill="#bbb" text-anchor="middle">75%</text>';
        html += '<text x="' + cx + '" y="' + (cy - r * 0.28) + '" font-size="8" fill="#bbb" text-anchor="middle">25%</text>';

        // 数据填充
        var scores = [Math.max(0.02, aiScore || 0), Math.max(0.02, dfScore || 0), Math.max(0.02, spScore || 0)];
        var dataPts = [];
        for (var d = 0; d < 3; d++) {
            dataPts.push((cx + (angles[d].x - cx) * scores[d]) + ',' + (cy + (angles[d].y - cy) * scores[d]));
        }
        html += '<polygon points="' + dataPts.join(' ') + '" fill="rgba(124,58,237,0.12)" stroke="#7c3aed" stroke-width="2" stroke-linejoin="round"/>';

        // 数据点
        var dotColors = ['#7c3aed', '#f59e0b', '#06b6d4'];
        for (var d = 0; d < 3; d++) {
            var dx = cx + (angles[d].x - cx) * scores[d];
            var dy = cy + (angles[d].y - cy) * scores[d];
            html += '<circle cx="' + dx + '" cy="' + dy + '" r="4" fill="' + dotColors[d] + '" stroke="#fff" stroke-width="1.5"/>';
        }

        // 标签
        var labels = ['AI全图生成', '换脸', '图像篡改'];
        var anchors = ['middle', 'end', 'start'];
        for (var d = 0; d < 3; d++) {
            var lx = cx + (angles[d].x - cx) * 1.22;
            var ly = cy + (angles[d].y - cy) * 1.16;
            html += '<text x="' + lx + '" y="' + ly + '" font-size="11" fill="#555" font-weight="600" text-anchor="' + anchors[d] + '">' + labels[d] + '</text>';
        }

        svg.innerHTML = html;
    }

    

    

    // ==================== 填充检测过程面板 ====================
    function populateProcessPanel() {
        if (!lastDetails) return;

        populateContentRiskPanel();

        // 水印短路：仅展示「检测到水印」提示，不展示算法结果
        if (lastShortcut) {
            setProcessSectionVisibility(false);
            populateWatermarkNotice();
            return;
        }

        // 正常流程：恢复算法区可见（各子函数按数据再行显隐）
        setProcessSectionVisibility(true);
        var c2pa = lastWatermark && lastWatermark.c2pa_verification;
        var metadataC2pa = lastWatermark && lastWatermark.metadata && lastWatermark.metadata.c2pa;
        if ((c2pa && c2pa.present) || (metadataC2pa && metadataC2pa.present)) {
            if (lastVisibleWatermark && lastVisibleWatermark.detected) populateWatermarkNotice(lastVisibleWatermark);
            else hideWatermarkNotice();
        }
        else if (lastWatermark && lastWatermark.detected) populateWatermarkNotice();
        else hideWatermarkNotice();
        var aiScore = (lastCombined && lastCombined.final_score) || 0;
        var dfScore = (lastDeepfakeCombined && lastDeepfakeCombined.final_score) || 0;
        var spScore = (lastTamperCombined && lastTamperCombined.final_score) || 0;

        // 1. 雷达图
        drawRadarChart(aiScore, dfScore, spScore);

        // 2. 进度条
        var items = [
            { id: 'ai', score: aiScore, available: !!lastCombined, name: 'AI全图生成检测', high: '图片整体由扩散模型/GAN 合成的可能性' },
            { id: 'df', score: dfScore, available: !!lastDeepfakeCombined, name: '深度换脸检测', high: '面部区域被 AI 替换/伪造的可能性' },
            { id: 'sp', score: spScore, available: !!lastTamperCombined, name: '图像篡改', high: '图片存在局部区域拼接、复制粘贴等篡改痕迹' }
        ];

        items.forEach(function (item) {
            var elScore = document.getElementById('proc-' + item.id + '-score');
            var elFill  = document.getElementById('proc-' + item.id + '-fill');
            var elDesc  = document.getElementById('proc-' + item.id + '-desc');
            if (!item.available) {
                if (elScore) elScore.textContent = '未检测';
                if (elFill) elFill.style.width = '0%';
                if (elDesc) { elDesc.textContent = '该项未执行'; elDesc.style.color = '#8592a6'; }
                return;
            }
            var pct = (item.score * 100).toFixed(1);

            if (elScore) elScore.textContent = pct + '分';
            if (elFill) {
                elFill.style.width = '0%';
                setTimeout(function (fill, w) { fill.style.width = w; }, 50, elFill, pct + '%');
            }
            if (elDesc) {
                if (item.score >= 0.7) {
                    elDesc.textContent = '高风险 — ' + item.high;
                    elDesc.style.color = '#dc2626';
                } else if (item.score >= 0.4) {
                    elDesc.textContent = '中等风险 — ' + item.high;
                    elDesc.style.color = '#d97706';
                } else {
                    elDesc.textContent = '低风险 — 未发现明显异常';
                    elDesc.style.color = '#16a34a';
                }
            }
            if (elFill) {
                if (item.score >= 0.7)      elFill.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
                else if (item.score >= 0.4) elFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
                else                         elFill.style.background = 'linear-gradient(90deg, #22c55e, #4ade80)';
            }
        });

        // 5. NPR 噪声模式分析
        populateNPRPanel();

        // 6. 专项预训练模型
        populateSpecializedModels();

        // 7. DeepFake 零模型专项检测
        var noFace = lastDeepfakeCombined && !lastDeepfakeCombined.face_detected;
        if (lastDeepfake && !noFace) {
            populateDeepfakeResults(lastDeepfake);
        } else {
            hideDeepfakeSections();
        }

        // 8. 图像篡改 零模型+纯模型 专项检测
        if (lastTamper) {
            populateTamperCombinedVerdict(lastTamperCombined);
            populateTamperResults(lastTamper);
        } else {
            hideTamperSections();
        }

        // 8. DeepFake 综合判定 + 纯模型
        if (lastDeepfakeCombined) {
            populateDFCombinedVerdict(lastDeepfakeCombined);
        }
        if (lastSpecialized && lastSpecialized.deepfake_detector && !noFace) {
            populateDFModelCard(lastSpecialized.deepfake_detector);
        } else {
            var dfmc2 = document.getElementById('df-model-card');
            if (dfmc2) dfmc2.style.display = 'none';
        }
    }

    // ==================== 图片内容识别 + 知识库风险研判 ====================
    function populateContentRiskPanel() {
        var section = document.getElementById('content-risk-section');
        if (!section) return;
        var analysis = lastContentAnalysis;
        if (!analysis || analysis.status === 'skipped') {
            section.style.display = 'none';
            return;
        }
        section.style.display = '';

        var scope = document.getElementById('content-risk-scope');
        var sceneLabel = analysis.crime_scene_label || (analysis.forensic_context || {}).crime_scene_label;
        if (scope) {
            scope.style.display = sceneLabel ? '' : 'none';
            scope.textContent = sceneLabel ? ('犯罪场景：' + sceneLabel) : '';
        }

        var level = analysis.risk_level || '未知';
        var levelEl = document.getElementById('content-risk-level');
        if (levelEl) {
            levelEl.textContent = level + '风险';
            levelEl.style.background = level === '高' ? '#dc2626' : level === '中' ? '#d97706' : level === '低' ? '#16a34a' : '#6b7280';
        }
        var desc = document.getElementById('content-risk-description');
        if (desc) desc.textContent = analysis.scene_description || analysis.detail || '内容识别服务暂不可用。';

        function fillChips(id, values, type) {
            var el = document.getElementById(id);
            if (!el) return;
            var list = values || [];
            if (!list.length) { el.innerHTML = '<span class="content-risk-chip">未识别</span>'; return; }
            el.innerHTML = list.map(function (item) {
                var label = typeof item === 'string' ? item : item.tag;
                var itemLevel = typeof item === 'string' ? '' : item.level;
                var cls = type === 'risk' ? (itemLevel === '高' ? ' risk-high' : itemLevel === '中' ? ' risk-medium' : ' risk-low') : '';
                return '<span class="content-risk-chip' + cls + '">' + escapeHtml(label || '未命名') + '</span>';
            }).join('');
        }
        fillChips('content-risk-tags', analysis.content_tags, 'content');
        fillChips('content-risk-risks', analysis.risk_tags, 'risk');
        fillChips('content-risk-knowledge', analysis.matched_knowledge, 'content');

        var evidence = document.getElementById('content-risk-evidence');
        if (evidence) evidence.textContent = (analysis.evidence || []).join('；') || '暂无可展示的识别依据。';
        var checks = document.getElementById('content-risk-checks');
        if (checks) checks.textContent = (analysis.recommended_checks || []).join('；') || '暂无建议核查项。';
        var ocr = document.getElementById('content-risk-ocr');
        if (ocr) {
            var texts = analysis.ocr_text || [];
            ocr.style.display = texts.length ? '' : 'none';
            ocr.textContent = texts.length ? 'OCR 识别文本：' + texts.join(' ｜ ') : '';
        }
    }

    // ==================== 水印短路面板（替代算法结果） ====================
    // 隐藏/恢复「综合分析比较」中的全部算法区块（雷达图、得分条、路径分隔线、各专项面板）
    function setProcessSectionVisibility(visible) {
        var show = visible ? '' : 'none';

        // 顶部雷达图 + 三条得分条
        var topRow = document.querySelector('.process-panel .process-top-row');
        if (topRow) topRow.style.display = show;
        var content = document.querySelector('.process-panel .process-content');
        if (content) content.style.display = show;

        // 专项明细折叠标题在水印快捷判定时也一并隐藏。
        ['ai-detail-group', 'df-detail-group', 'tp-detail-group'].forEach(function (id) {
            var group = document.getElementById(id);
            if (group) group.style.display = show;
        });

        // 各专项区块
        var ids = [
            'combined-section', 'npr-section', 'neural-analysis-section',
            'df-combined-section', 'df-ela-section', 'df-model-card',
            'tp-combined-section', 'tp-forensic-section', 'tp-pf-section'
        ];
        ids.forEach(function (id) {
            var el = document.getElementById(id);
            if (el && !visible) el.style.display = 'none';
        });

        // 路径分隔线（三条专项检测标题）
        var dividers = document.querySelectorAll('.process-panel .path-divider');
        dividers.forEach(function (d) { d.style.display = show; });
    }

    function hideWatermarkNotice() {
        var notice = document.getElementById('watermark-notice');
        if (notice) notice.style.display = 'none';
    }

    function setWn(id, text) {
        var el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    // 填充水印快速判定提示卡片
    function populateWatermarkNotice(watermarkOverride) {
        var notice = document.getElementById('watermark-notice');
        if (!notice) return;

        var wm = watermarkOverride || lastWatermark || {};
        var details = lastDetails || {};
        var c2pa = wm.c2pa_verification || {};
        var metadataC2pa = wm.metadata && wm.metadata.c2pa;
        if (c2pa.present || (metadataC2pa && metadataC2pa.present)) {
            notice.style.display = 'none';
            return;
        }
        notice.style.display = '';

        var source = wm.source || details.watermark_source || '未知平台';
        var confidence = (wm.confidence !== undefined && wm.confidence !== null)
            ? wm.confidence : details.watermark_confidence;
        var position = wm.position || null;
        var text = wm.detected_text || null;
        var detail = wm.detail || details.verdict_detail || '';
        setWn('wn-title', '检测到 AI 生成水印标识');
        setWn('wn-subtitle', '水印线索参与检测结果研判');
        setWn('wn-method', watermarkOverride ? '可见水印' : (lastWatermarkIsHidden ? '隐式元数据标识' : '可见水印'));

        setWn('wn-source', source);
        setWn('wn-confidence', (confidence !== undefined && confidence !== null)
            ? (confidence * 100).toFixed(1) + '%' : '--');

        var posRow = document.getElementById('wn-position-row');
        var posEl = document.getElementById('wn-position');
        if (posRow) posRow.style.display = position ? '' : 'none';
        if (posEl) posEl.textContent = position || '--';

        var txtRow = document.getElementById('wn-text-row');
        var txtEl = document.getElementById('wn-text');
        if (txtRow) txtRow.style.display = text ? '' : 'none';
        if (txtEl) txtEl.textContent = text || '--';

        var detailEl = document.getElementById('wn-detail');
        if (detailEl) detailEl.textContent = detail || '';
    }

    // ==================== 图像物理分析面板（NPR + CFA/传感器物证） ====================
    function populateNPRPanel() {
        var nprSection = document.getElementById('npr-section');
        if (!nprSection || !lastNpr) {
            if (nprSection) nprSection.style.display = 'none';
            return;
        }
        nprSection.style.display = '';

        var nprScore = lastNpr.score || 0;
        var verdict = lastNpr.verdict || 'error';
        var features = lastNpr.features || {};

        // 1. 圆形仪表盘
        var gaugeFill = document.getElementById('npr-gauge-fill');
        var gaugeScore = document.getElementById('npr-gauge-score');
        var gaugeLabel = document.getElementById('npr-gauge-label');

        if (gaugeFill) {
            var circumference = 2 * Math.PI * 50; // r=50
            var offset = circumference * (1 - nprScore);
            gaugeFill.style.strokeDasharray = circumference;
            gaugeFill.style.strokeDashoffset = offset;
            if (nprScore >= 0.60) {
                gaugeFill.style.stroke = '#ef4444';
            } else if (nprScore >= 0.35) {
                gaugeFill.style.stroke = '#f59e0b';
            } else {
                gaugeFill.style.stroke = '#22c55e';
            }
        }
        if (gaugeScore) gaugeScore.textContent = (nprScore * 100).toFixed(0);
        if (gaugeLabel) {
            var labelMap = { ai_likely: '疑似AI生成', real_likely: '疑似真实', uncertain: '不确定', error: '分析失败' };
            gaugeLabel.textContent = labelMap[verdict] || verdict;
            gaugeLabel.style.color = verdict === 'ai_likely' ? '#dc2626' : verdict === 'real_likely' ? '#16a34a' : '#888';
        }

        // 2. 四项特征条
        var featDefs = [
            { key: 'peak_ratio',       id: 'nf-peak',      fillId: 'nf-peak-fill',      name: '峰值密度' },
            { key: 'spectral_flatness', id: 'nf-flatness',  fillId: 'nf-flatness-fill',  name: '频谱平坦度' },
            { key: 'high_freq_ratio',  id: 'nf-hf',        fillId: 'nf-hf-fill',        name: '高频能量比' },
            { key: 'radial_variance',  id: 'nf-variance',  fillId: 'nf-variance-fill',   name: '径向方差' },
        ];

        featDefs.forEach(function (fd) {
            var elVal = document.getElementById(fd.id);
            var elFill = document.getElementById(fd.fillId);
            var raw = features[fd.key];
            if (elVal && elFill && raw !== undefined) {
                // 归一化到 0~1 展示
                var norm;
                if (fd.key === 'peak_ratio')       norm = Math.min(raw * 500, 1);
                else if (fd.key === 'spectral_flatness') norm = raw;
                else if (fd.key === 'high_freq_ratio')  norm = Math.min(raw * 5, 1);
                else if (fd.key === 'radial_variance')  norm = Math.min(raw * 30, 1);
                else norm = raw;
                elVal.textContent = (norm * 100).toFixed(1) + '%';
                elFill.style.width = (norm * 100).toFixed(0) + '%';
                if (norm >= 0.7)      elFill.style.background = '#ef4444';
                else if (norm >= 0.4) elFill.style.background = '#f59e0b';
                else                  elFill.style.background = '#22c55e';
            }
        });

        // 3. 相机物理层维度：这些是已纳入 NPR 综合得分的内部评分。
        var physical = lastNpr.physical_dimensions || {};
        var physicalDefs = [
            { value: lastNpr.raw_noise_score, id: 'pf-npr', fillId: 'pf-npr-fill' },
            { value: physical.noise_weakness, id: 'pf-noise', fillId: 'pf-noise-fill' },
            { value: physical.cfa_weakness, id: 'pf-cfa', fillId: 'pf-cfa-fill' },
            { value: physical.camera_trace_weakness, id: 'pf-camera', fillId: 'pf-camera-fill' },
            { value: physical.metadata_weakness, id: 'pf-meta', fillId: 'pf-meta-fill' }
        ];
        physicalDefs.forEach(function (fd) {
            var valEl = document.getElementById(fd.id);
            var fillEl = document.getElementById(fd.fillId);
            if (!valEl || !fillEl) return;
            if (fd.value === undefined || fd.value === null) {
                valEl.textContent = '--';
                fillEl.style.width = '0%';
                fillEl.classList.add('nf-fill-neutral');
                return;
            }
            var value = Math.max(0, Math.min(1, Number(fd.value) || 0));
            valEl.textContent = (value * 100).toFixed(1) + '%';
            fillEl.style.width = (value * 100).toFixed(0) + '%';
            fillEl.classList.remove('nf-fill-neutral');
            fillEl.style.background = value >= 0.7 ? '#ef4444' : value >= 0.4 ? '#f59e0b' : '#22c55e';
        });

        // 4. 与 GPU 结果交叉验证
        var crossEl = document.getElementById('npr-cross-verdict');
        var ncvIcon = document.getElementById('ncv-icon');
        var ncvText = document.getElementById('ncv-text');
        if (crossEl && lastDetails) {
            var gpuAiScore = lastDetails.ai_generated || 0;

            var nprFlag = nprScore >= 0.60;
            var gpuFlag = gpuAiScore >= 0.5;

            if (nprFlag && gpuFlag) {
                ncvIcon.textContent = '✅';
                ncvText.textContent = '图像物理分析与 GPU 一致：两路均判定为 AI 生成，置信度高';
                crossEl.className = 'npr-cross-verdict npr-agree';
            } else if (!nprFlag && !gpuFlag) {
                ncvIcon.textContent = '✅';
                ncvText.textContent = '图像物理分析与 GPU 一致：两路均判定为真实，可信度高';
                crossEl.className = 'npr-cross-verdict npr-agree';
            } else if (nprFlag && !gpuFlag) {
                ncvIcon.textContent = '⚠️';
                ncvText.textContent = '图像物理分析判为 AI 生成但 GPU 判为真实 — 建议人工复核';
                crossEl.className = 'npr-cross-verdict npr-conflict';
            } else {
                ncvIcon.textContent = '⚠️';
                ncvText.textContent = 'GPU 判为 AI 生成但图像物理分析判为真实 — 可能存在新型生成算法';
                crossEl.className = 'npr-cross-verdict npr-conflict';
            }
        }
    }

    // ==================== 专项预训练模型面板 ====================
    function populateSpecializedModels() {
        var neuralSection = document.getElementById('neural-analysis-section');
        if (!lastSpecialized) {
            hideSM('neural-analysis-section');
            return;
        }

        var hasAvailableModel = false;

        var modelRows = [
            { key: 'univfd', rowId: 'neural-univfd-row', scoreId: 'neural-univfd-score', fillId: 'neural-univfd-fill' },
            { key: 'probe_dinov2', rowId: 'neural-probe-row', scoreId: 'neural-probe-score', fillId: 'neural-probe-fill' }
        ];
        var weights = { univfd: 0.20, probe_dinov2: 0.30 };
        var weightedScore = 0;
        var weightTotal = 0;

        modelRows.forEach(function (item) {
            var model = lastSpecialized[item.key];
            var available = model && model.available !== false && typeof model.ai_score === 'number';
            var row = document.getElementById(item.rowId);
            if (row) row.style.display = available ? '' : 'none';
            if (!available) return;

            hasAvailableModel = true;
            var score = Math.max(0, Math.min(1, model.ai_score));
            var scoreEl = document.getElementById(item.scoreId);
            var fillEl = document.getElementById(item.fillId);
            if (scoreEl) scoreEl.textContent = (score * 100).toFixed(1) + '%';
            if (fillEl) {
                fillEl.style.width = (score * 100).toFixed(0) + '%';
                fillEl.style.background = score >= 0.7 ? '#ef4444' : score >= 0.4 ? '#f59e0b' : '#22c55e';
            }
            weightedScore += score * weights[item.key];
            weightTotal += weights[item.key];
        });

        if (neuralSection) neuralSection.style.display = hasAvailableModel ? '' : 'none';
        if (!hasAvailableModel) return;

        var neuralScore = weightTotal ? weightedScore / weightTotal : 0;
        var gaugeFill = document.getElementById('neural-gauge-fill');
        var gaugeScore = document.getElementById('neural-gauge-score');
        var gaugeLabel = document.getElementById('neural-gauge-label');
        if (gaugeFill) {
            var circumference = 2 * Math.PI * 50;
            gaugeFill.style.strokeDasharray = circumference;
            gaugeFill.style.strokeDashoffset = circumference * (1 - neuralScore);
            gaugeFill.style.stroke = neuralScore >= 0.60 ? '#ef4444' : '#22c55e';
        }
        if (gaugeScore) gaugeScore.textContent = (neuralScore * 100).toFixed(0);
        if (gaugeLabel) gaugeLabel.textContent = neuralScore >= 0.60 ? '疑似 AI 生成' : '倾向真实';

    }

    function showSM(id) {
        var el = document.getElementById(id);
        if (el) el.style.display = '';
    }

    function hideSM(id) {
        var el = document.getElementById(id);
        if (el) el.style.display = 'none';
    }

    // 深度换脸等其余专项面板共用的概率进度条渲染。
    function setBar(id, score, type) {
        var el = document.getElementById(id);
        if (!el) return;
        var value = Math.max(0, Math.min(1, Number(score) || 0));
        el.style.width = (value * 100).toFixed(0) + '%';
        el.style.background = type === 'ai' ? '#ef4444' : '#22c55e';
    }

    // ==================== 图像物理分析 + 神经网络分析融合判定展示 ====================
    function populateCombinedVerdict(combined) {
        if (!combined) {
            var cs = document.getElementById('combined-section');
            if (cs) cs.style.display = 'none';
            return;
        }
        var cs = document.getElementById('combined-section');
        if (cs) cs.style.display = '';

        var badgeEl = document.getElementById('combined-badge');
        if (badgeEl) badgeEl.textContent = combined.fusion_mode || '图像物理分析 + 神经网络分析';
        var subtitleEl = document.getElementById('combined-subtitle');
        if (subtitleEl) {
            subtitleEl.textContent = '融合图像物理分析与可用神经网络模型的多路判断';
        }

        // 综合得分
        var scoreEl = document.getElementById('combined-score');
        var finalPoints = (Math.max(0, Math.min(1, Number(combined.final_score) || 0)) * 100).toFixed(1);
        if (scoreEl) scoreEl.textContent = finalPoints;

        // 综合判定
        var verdictEl = document.getElementById('combined-verdict');
        if (verdictEl) {
            verdictEl.textContent = combined.final_verdict;
            verdictEl.className = 'combined-verdict ' + getVerdictClass(combined.final_score);
        }

        // 分解贡献
        var nprEl = document.getElementById('cb-npr');
        if (nprEl) nprEl.textContent = ((combined.npr_contribution || 0) * 100).toFixed(1) + ' 分';

        var neuralEl = document.getElementById('cb-neural');
        if (neuralEl) {
            var neuralContribution = combined.neural_contribution;
            if (neuralContribution === undefined || neuralContribution === null) {
                neuralContribution = (combined.univfd_contribution || 0)
                    + (combined.probe_contribution || 0);
            }
            neuralEl.textContent = (neuralContribution * 100).toFixed(1) + ' 分';
        }

        var totalEl = document.getElementById('cb-total');
        if (totalEl) totalEl.textContent = finalPoints + ' 分';
    }

    function getVerdictClass(score) {
        return score >= 0.60 ? 'cv-ai' : 'cv-real';
    }

    // ==================== DeepFake 专项检测结果展示 ====================

    function hideDeepfakeSections() {
        ['df-combined-section', 'df-ela-section'].forEach(function (id) {
            var el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
    }

    function populateDeepfakeResults(df) {
        if (!df) {
            hideDeepfakeSections();
            return;
        }

        // ── ELA 结果 ──
        if (df.ela) {
            populateELAResult(df.ela);
        } else {
            var elaSec = document.getElementById('df-ela-section');
            if (elaSec) elaSec.style.display = 'none';
        }

    }

    // ── DeepFake 综合判定环（服务器端融合的 deepfake_combined）──
    function populateDFCombinedVerdict(combined) {
        var cs = document.getElementById('df-combined-section');
        if (!cs) return;
        cs.style.display = '';

        var score = combined.final_score || 0;
        var finalPoints = (Math.max(0, Math.min(1, Number(score) || 0)) * 100).toFixed(1);
        var scoreEl = document.getElementById('df-combined-score');
        if (scoreEl) scoreEl.textContent = finalPoints;

        var verdictEl = document.getElementById('df-combined-verdict');
        if (verdictEl) {
            verdictEl.textContent = combined.final_verdict || '--';
            verdictEl.className = 'combined-verdict ' + (score > 0.60 ? 'cv-ai' : 'cv-real');
        }

        // 得分贡献
        var zeroEl = document.getElementById('df-cb-zero');
        if (zeroEl) zeroEl.textContent = ((combined.zero_contribution || 0) * 100).toFixed(1) + ' 分';

        var modelEl2 = document.getElementById('df-cb-model');
        if (modelEl2) modelEl2.textContent = ((combined.model_contribution || 0) * 100).toFixed(1) + ' 分';

        var totalEl = document.getElementById('df-cb-total');
        if (totalEl) totalEl.textContent = finalPoints + ' 分';
    }

    // ── ELA 误差分析展示 ──
    function populateELAResult(ela) {
        var elaSec = document.getElementById('df-ela-section');
        if (elaSec) elaSec.style.display = '';

        // Gauge
        var gaugeFill = document.getElementById('df-ela-gauge-fill');
        var gaugeScore = document.getElementById('df-ela-gauge-score');
        var gaugeLabel = document.getElementById('df-ela-gauge-label');

        var elaCirc = 2 * Math.PI * 50;
        if (gaugeFill) {
            gaugeFill.style.strokeDasharray = elaCirc;
            gaugeFill.style.strokeDashoffset = elaCirc - (ela.score * elaCirc);
        }
        if (gaugeScore) gaugeScore.textContent = (ela.score * 100).toFixed(0);
        if (gaugeLabel) {
            var elaLabels = {
                'fake_likely': '疑似篡改',
                'real_likely': '正常',
                'uncertain': '存疑',
                'error': '异常'
            };
            gaugeLabel.textContent = elaLabels[ela.verdict] || ela.verdict;
        }

        // Features bars
        setFeatureBarDF('df-ela-anomaly', 'df-ela-anomaly-fill', (ela.anomaly_ratio || 0) * 100, 'ai');
        setFeatureBarDF('df-ela-cluster', 'df-ela-cluster-fill', (ela.cluster_ratio || 0) * 100, 'ai');
        setFeatureBarDF('df-ela-mean', 'df-ela-mean-fill', (ela.mean_error || 0) / 20 * 100, 'neutral');
        setFeatureBarDF('df-ela-std', 'df-ela-std-fill', (ela.std_error || 0) / 30 * 100, 'neutral');

        setFeatureVal('df-ela-anomaly', ((ela.anomaly_ratio || 0) * 100).toFixed(2) + '%');
        setFeatureVal('df-ela-cluster', ((ela.cluster_ratio || 0) * 100).toFixed(2) + '%');
        setFeatureVal('df-ela-mean', (ela.mean_error || 0).toFixed(2));
        setFeatureVal('df-ela-std', (ela.std_error || 0).toFixed(2));
    }

    // ── DeepFake 专用辅助函数 ──

    function setFeatureBarDF(labelId, fillId, percent, type) {
        void labelId;
        var fillEl = document.getElementById(fillId);
        if (!fillEl) return;

        var capped = Math.min(Math.max(percent, 0), 100);
        fillEl.style.width = capped.toFixed(0) + '%';

        if (type === 'ai') {
            fillEl.style.background = '#ef4444';
        } else if (type === 'neutral') {
            fillEl.style.background = '#888';
        } else if (type === 'compare') {
            fillEl.style.background = '#6366f1';
        } else if (type === 'compare_bg') {
            fillEl.style.background = '#a78bfa';
        } else if (type === 'warn') {
            fillEl.style.background = capped > 65 ? '#ef4444' : capped > 35 ? '#f59e0b' : '#22c55e';
        }
    }

    function setFeatureVal(id, text) {
        var el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    // ==================== DeepFake ViT 纯模型卡片 ====================
    function populateDFModelCard(modelResult) {
        var card = document.getElementById('df-model-card');
        if (!card) return;
        card.style.display = '';

        var fakeScore = modelResult.deepfake_score || 0;
        var realScore = modelResult.real_score || 0;

        // DeepFake 概率
        var fakeEl = document.getElementById('df-model-fake');
        if (fakeEl) fakeEl.textContent = (fakeScore * 100).toFixed(1) + '%';
        setBar('df-model-fake-fill', fakeScore, 'ai');

        // 真实概率
        var realEl = document.getElementById('df-model-real');
        if (realEl) realEl.textContent = (realScore * 100).toFixed(1) + '%';
        setBar('df-model-real-fill', realScore, 'human');

    }

    // ==================== 图像篡改 专项检测结果展示 ====================

    function hideTamperSections() {
        ['tp-combined-section', 'tp-forensic-section', 'tp-pf-section'].forEach(function (id) {
            var el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
    }

    function populateTamperResults(tp) {
        if (!tp) {
            hideTamperSections();
            return;
        }
        var sec = document.getElementById('tp-combined-section');
        if (sec) sec.style.display = '';

        var forensicSec = document.getElementById('tp-forensic-section');
        if (forensicSec) forensicSec.style.display = 'none';

        // ── TruFor 主模型 ──
        if (tp.trufor) {
            populatePatchFeatureResult(tp.trufor || {});
        } else {
            var pfSec = document.getElementById('tp-pf-section');
            if (pfSec) pfSec.style.display = 'none';
        }
    }

    // ── 图像篡改 融合判定展示 ──
    function populateTamperCombinedVerdict(combined) {
        if (!combined) {
            var cs = document.getElementById('tp-combined-section');
            if (cs) cs.style.display = 'none';
            return;
        }
        var cs = document.getElementById('tp-combined-section');
        if (cs) cs.style.display = '';

        // 综合得分
        var finalPoints = (Math.max(0, Math.min(1, Number(combined.final_score) || 0)) * 100).toFixed(1);
        var scoreEl = document.getElementById('tp-combined-score');
        if (scoreEl) scoreEl.textContent = finalPoints;

        // 判定
        var verdictEl = document.getElementById('tp-combined-verdict');
        if (verdictEl) {
            verdictEl.textContent = combined.final_verdict || '--';
            if (combined.final_score >= 0.60) {
                verdictEl.style.color = '#dc2626';
            } else {
                verdictEl.style.color = '#16a34a';
            }
        }

        // 得分贡献
        var pfEl = document.getElementById('tp-cb-pf');
        if (pfEl) pfEl.textContent = combined.model_score == null ? '不可用' : ((combined.model_score || 0) * 100).toFixed(1) + ' 分';

        var totalEl = document.getElementById('tp-cb-total');
        if (totalEl) totalEl.textContent = finalPoints + ' 分';
    }

    // ── 图像篡改物理分析展示（复制-移动 + 区块噪声）──
    function populateTamperForensicsResult(cm, bn) {
        var forensicSec = document.getElementById('tp-forensic-section');
        if (forensicSec) forensicSec.style.display = '';

        var cmScore = Math.max(0, Math.min(1, Number(cm.score) || 0));
        var bnScore = Math.max(0, Math.min(1, Number(bn.score) || 0));
        // 仅作该物理证据面板的等权汇总；不会改变后端篡改综合判定的算法与权重。
        var score = (cmScore + bnScore) / 2;
        var gaugeFill = document.getElementById('tp-forensic-gauge-fill');
        var gaugeScore = document.getElementById('tp-forensic-gauge-score');
        var gaugeLabel = document.getElementById('tp-forensic-gauge-label');
        var circ = 2 * Math.PI * 50;
        if (gaugeFill) {
            gaugeFill.style.strokeDasharray = circ;
            gaugeFill.style.strokeDashoffset = circ - (score * circ);
            gaugeFill.style.stroke = score >= 0.7 ? '#ef4444' : (score >= 0.5 ? '#f59e0b' : '#22c55e');
        }
        if (gaugeScore) gaugeScore.textContent = (score * 100).toFixed(0);
        if (gaugeLabel) gaugeLabel.textContent = score >= 0.7 ? '疑似篡改' : (score >= 0.5 ? '存疑' : '未见异常');

        var maxCount = 200;
        setFeatureBarDF('tp-cm-count', 'tp-cm-count-fill', Math.min((Number(cm.match_count) || 0) / maxCount * 100, 100), 'neutral');
        setFeatureBarDF('tp-cm-density', 'tp-cm-density-fill', (Number(cm.match_density) || 0) * 400, 'warn');
        setFeatureVal('tp-cm-count', Number(cm.match_count) || 0);
        setFeatureVal('tp-cm-density', (Number(cm.match_density) || 0).toFixed(4));

        setFeatureBarDF('tp-bn-disp', 'tp-bn-disp-fill', (Number(bn.dispersion) || 0) * 500, 'warn');
        setFeatureBarDF('tp-bn-mean', 'tp-bn-mean-fill', (Number(bn.mean_score) || 0) * 100, 'neutral');
        setFeatureVal('tp-bn-disp', (Number(bn.dispersion) || 0).toFixed(4));
        setFeatureVal('tp-bn-mean', (Number(bn.mean_score) || 0).toFixed(3));
    }

    // ── TruFor 深度篡改定位展示 ──
    function populatePatchFeatureResult(pf) {
        var pfSec = document.getElementById('tp-pf-section');
        if (pfSec) pfSec.style.display = '';

        var gaugeFill = document.getElementById('tp-pf-gauge-fill');
        var gaugeScore = document.getElementById('tp-pf-gauge-score');
        var gaugeLabel = document.getElementById('tp-pf-gauge-label');

        var available = pf.available === true;
        var score = Math.max(0, Math.min(1, Number(pf.score) || 0));
        var reliability = Math.max(0, Math.min(1, Number(pf.reliability) || 0));
        var circ = 2 * Math.PI * 50;
        if (gaugeFill) {
            gaugeFill.style.strokeDasharray = circ;
            gaugeFill.style.strokeDashoffset = circ - (score * circ);
        }
        if (gaugeScore) gaugeScore.textContent = available ? (score * 100).toFixed(0) : '--';
        if (gaugeLabel) gaugeLabel.textContent = available ? 'TruFor 已完成' : 'TruFor 不可用';

        var coverage = Math.max(0, Math.min(1, Number(pf.map_coverage) || 0));
        setFeatureBarDF('tp-pf-mean-sim', 'tp-pf-mean-sim-fill', score * 100, 'warn');
        setFeatureBarDF('tp-pf-min-sim', 'tp-pf-min-sim-fill', reliability * 100, 'neutral');
        setFeatureBarDF('tp-pf-anomaly', 'tp-pf-anomaly-fill', coverage * 100, 'warn');
        setFeatureVal('tp-pf-mean-sim', available ? score.toFixed(4) : '--');
        setFeatureVal('tp-pf-min-sim', available ? reliability.toFixed(4) : '--');
        setFeatureVal('tp-pf-anomaly', available ? (coverage * 100).toFixed(1) + '%' : '--');

        var mapBox = document.getElementById('tp-trufor-maps');
        var map = document.getElementById('tp-trufor-map');
        var conf = document.getElementById('tp-trufor-confidence');
        var truforMapBlock = document.getElementById('tp-trufor-map-block');
        var truforConfidenceBlock = document.getElementById('tp-trufor-confidence-block');
        var hasTruForMaps = available && pf.map_png && pf.confidence_png;
        if (hasTruForMaps) {
            if (map) map.src = pf.map_png;
            if (conf) conf.src = pf.confidence_png;
            if (truforMapBlock) truforMapBlock.style.display = hasTruForMaps ? '' : 'none';
            if (truforConfidenceBlock) truforConfidenceBlock.style.display = hasTruForMaps ? '' : 'none';
            if (mapBox) mapBox.style.display = '';
        } else if (mapBox) {
            mapBox.style.display = 'none';
        }
    }

    // ========== SSE 流式检测（图像专用，含进度条） ==========

    function startImageDetection(formData) {
        detectBtn.disabled = true;
        detectBtn.textContent = '检测中...';
        resetProgressUI();
        progressOverlay.style.display = 'flex';

        fetch('/api/detect/image/stream', {
            method: 'POST',
            body: formData
        }).then(function (response) {
            if (!response.ok) {
                return response.text().then(function (body) {
                    var message = '检测接口请求失败（HTTP ' + response.status + '），请稍后重试';
                    try {
                        var errorData = JSON.parse(body);
                        message = errorData.msg || errorData.message || message;
                    } catch (error) {}
                    showProgressError(message);
                });
            }
            if (!response.body) {
                showProgressError('检测接口未返回数据，请重试');
                return;
            }
            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = '';
            var streamEnded = false;

            function readStream() {
                return reader.read().then(function (chunk) {
                    if (chunk.done) {
                        if (!streamEnded) {
                            showProgressError('检测连接已断开，未收到完整结果，请重试');
                        }
                        return;
                    }
                    buffer += decoder.decode(chunk.value, { stream: true });
                    var parts = buffer.split('\n\n');
                    buffer = parts.pop() || '';
                    parts.forEach(function (block) {
                        if (!block.trim()) return;
                        var eventType = '';
                        var eventData = '';
                        block.split('\n').forEach(function (line) {
                            if (line.indexOf('event: ') === 0) eventType = line.slice(7).trim();
                            else if (line.indexOf('data: ') === 0) eventData = line.slice(6);
                        });
                        if (eventType && eventData) {
                            try {
                                handleSSEEvent(eventType, JSON.parse(eventData));
                                if (eventType === 'done' || eventType === 'error') streamEnded = true;
                            }
                            catch (e) { console.warn('SSE解析失败:', e); }
                        }
                    });
                    return readStream();
                }).catch(function (err) {
                    if (!streamEnded) showProgressError('检测服务连接中断，请重试');
                });
            }
            return readStream();
        }).catch(function (err) {
            showProgressError('无法连接本地检测服务，请确认服务已启动后重试');
        });
    }

    function handleSSEEvent(type, data) {
        switch (type) {
            case 'progress': updateProgressUI(data); break;
            case 'result': _pendingResult = data; break;
            case 'done':
                if (_pendingResult) finishDetection(_pendingResult);
                else showProgressError('检测结束但未收到结果，请重试');
                break;
            case 'error': showProgressError(data.message || '检测服务返回错误'); break;
        }
    }

    var _pendingResult = null;

    function stopProgressTimer() {
        if (progressTimer) clearInterval(progressTimer);
        progressTimer = null;
    }

    function progressTimeText() {
        var seconds = Math.max(0, Math.floor((Date.now() - progressStartedAt) / 1000));
        return String(Math.floor(seconds / 60)).padStart(2, '0') + ':' + String(seconds % 60).padStart(2, '0');
    }

    function appendProgressActivity(message) {
        if (!message || message === progressLastMessage) return;
        progressLastMessage = message;
        var item = document.createElement('li');
        var time = document.createElement('span');
        time.className = 'progress-log-time';
        time.textContent = progressTimeText();
        var text = document.createElement('span');
        text.textContent = message;
        item.appendChild(time);
        item.appendChild(text);
        progressActivityList.appendChild(item);
        while (progressActivityList.children.length > 3) progressActivityList.firstElementChild.remove();
    }

    function updateProgressUI(data) {
        var complete = data.stepName === '完成';
        progressValue = Math.max(progressValue, Math.min(100, Math.max(0, Number(data.percent) || 0)));
        progressBarFill.style.width = progressValue + '%';
        progressPercent.innerHTML = Math.round(progressValue) + '<span>%</span>';
        progressBar.setAttribute('aria-valuenow', Math.round(progressValue));
        var titles = { '水印检测': '正在核验水印与平台标识', 'GPU推理': '正在进行模型推理', 'AI全图生成': '正在分析 AI 生成痕迹', 'DeepFake': '正在分析人脸伪造痕迹', '图像篡改': '正在检查图像篡改痕迹', '融合判定': '正在汇总鉴定证据', '水印判定': '正在汇总鉴定证据', '内容研判': '正在研判图片内容' };
        progressTitleText.textContent = complete ? '图像鉴定完成' : (titles[data.stepName] || '正在分析图片');
        progressSubText.textContent = [data.message, data.detail].filter(Boolean).join(' · ') || '正在处理，请稍候…';
        appendProgressActivity(data.message || data.detail);

        var stageMap = { '水印检测': 'watermark', 'GPU推理': 'gpu', 'AI全图生成': 'ai', 'DeepFake': 'deepfake', '图像篡改': 'tamper', '融合判定': 'fusion', '水印判定': 'fusion', '内容研判': 'content' };
        var states = data.stages || {};
        var statusText = { pending: '等待中', active: '分析中', done: '已完成', skipped: '已跳过', warning: '部分不可用', error: '已中断' };
        var handled = 0;
        var skipped = 0;
        var stepDots = progressStepsRow.querySelectorAll('.progress-step-dot');
        stepDots.forEach(function (dot) {
            var key = dot.getAttribute('data-stage');
            var state = states[key] || (dot.classList.contains('done') ? 'done' : 'pending');
            if (!data.stages) {
                if (dot.classList.contains('active')) state = 'done';
                if (key === stageMap[data.stepName]) state = 'active';
                if (complete && state === 'pending') state = 'skipped';
            }
            if (!statusText[state]) state = 'pending';
            dot.classList.remove('active', 'done', 'skipped', 'warning', 'error');
            if (state !== 'pending') dot.classList.add(state);
            dot.querySelector('.pstep-circle').textContent = state === 'done' ? '✓' : state === 'skipped' ? '−' : state === 'warning' || state === 'error' ? '!' : dot.getAttribute('data-step');
            dot.querySelector('.pstep-status').textContent = statusText[state];
            dot.setAttribute('aria-label', dot.querySelector('.pstep-label').textContent + '：' + statusText[state]);
            if (state === 'done' || state === 'skipped' || state === 'warning') handled++;
            if (state === 'skipped') skipped++;
        });
        progressStageCount.textContent = '已处理 ' + handled + ' / 7 项' + (skipped ? ' · 跳过 ' + skipped + ' 项' : '');
        var icons = { '水印检测': '⌕', 'GPU推理': '◈', 'AI全图生成': '✧', 'DeepFake': '◎', '图像篡改': '▧', '融合判定': '≋', '水印判定': '≋', '内容研判': '⌕', '完成': '✓' };
        progressStepIcon.textContent = icons[data.stepName] || '◎';
        if (complete) {
            stopProgressTimer();
            progressModal.classList.add('is-complete');
            progressStatus.textContent = '鉴定完成';
            progressFootnote.textContent = '分析已完成，正在展示鉴定结果';
        }
    }

    function showProgressError(msg) {
        progressErrorMsg.textContent = msg;
        progressErrorMsg.style.display = 'block';
        progressRetryBtn.style.display = 'block';
        stopProgressTimer();
        progressModal.classList.add('is-error');
        progressStatus.textContent = '检测中断';
        progressTitleText.textContent = '鉴定暂时中断';
        progressSubText.textContent = '检测未完成，可以重试';
        progressFootnote.textContent = '当前进度已保留，重试后将重新开始鉴定';
        progressStepIcon.textContent = '!';
        progressStepsRow.querySelectorAll('.active').forEach(function (dot) {
            dot.classList.remove('active');
            dot.classList.add('error');
            dot.querySelector('.pstep-circle').textContent = '!';
            dot.querySelector('.pstep-status').textContent = '已中断';
            dot.setAttribute('aria-label', dot.querySelector('.pstep-label').textContent + '：已中断');
        });
        appendProgressActivity('检测中断：' + msg);
        progressRetryBtn.focus();
        // 失败后必须恢复按钮状态；否则 requestSubmit 会被 submit 处理器直接拦截。
        if (detectBtn) {
            detectBtn.disabled = false;
            detectBtn.textContent = '开始检测';
        }
        progressRetryBtn.onclick = function () {
            // 直接按当前文件重新构造请求，避免依赖已禁用按钮的表单提交逻辑。
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                showProgressError('原始图片已丢失，请重新选择图片后再检测');
                return;
            }
            startImageDetection(new FormData(form));
        };
    }

    function resetProgressUI() {
        stopProgressTimer();
        progressStartedAt = Date.now();
        progressValue = 0;
        progressLastMessage = '';
        progressPreviousFocus = document.activeElement;
        progressModal.classList.remove('is-complete', 'is-error');
        progressStatus.textContent = '正在分析';
        progressPercent.innerHTML = '0<span>%</span>';
        progressBar.setAttribute('aria-valuenow', '0');
        progressStageCount.textContent = '准备分析';
        progressFootnote.textContent = '正在逐项核验图片，分析完成后将自动展示鉴定结果';
        progressActivityList.textContent = '';
        appendProgressActivity('正在准备图片与检测模型');
        progressElapsed.textContent = '已用时 0 秒';
        progressTimer = setInterval(function () {
            var seconds = Math.floor((Date.now() - progressStartedAt) / 1000);
            progressElapsed.textContent = '已用时 ' + (seconds < 60 ? seconds + ' 秒' : Math.floor(seconds / 60) + ' 分 ' + seconds % 60 + ' 秒');
        }, 1000);
        progressBarFill.style.width = '0%';
        progressTitleText.textContent = '准备开始鉴定';
        progressSubText.textContent = '正在连接检测服务，请稍候…';
        progressStepIcon.textContent = '◎';
        progressErrorMsg.style.display = 'none';
        progressRetryBtn.style.display = 'none';
        if (uploadPreviewUrl) {
            progressImage.src = uploadPreviewUrl;
            progressImage.style.display = '';
        } else {
            progressImage.removeAttribute('src');
            progressImage.style.display = 'none';
        }
        _pendingResult = null;
        var stepDots = progressStepsRow.querySelectorAll('.progress-step-dot');
        stepDots.forEach(function (dot) {
            dot.classList.remove('active', 'done', 'skipped', 'warning', 'error');
            dot.querySelector('.pstep-circle').textContent = dot.getAttribute('data-step');
            dot.querySelector('.pstep-status').textContent = '等待中';
            dot.setAttribute('aria-label', dot.querySelector('.pstep-label').textContent + '：等待中');
        });
    }

    function finishDetection(resultData) {
        stopProgressTimer();
        setTimeout(function () {
            progressOverlay.style.display = 'none';
            detectBtn.disabled = false;
            detectBtn.textContent = '开始检测';
            if (resultData) { showResult(resultData); }
            else { alert('检测完成但未获取到结果数据'); }
            if (progressPreviousFocus && document.contains(progressPreviousFocus)) progressPreviousFocus.focus();
        }, 800);
    }

        function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

// ════════════════════════════════════════════════
    // 批量检测模块
    // ════════════════════════════════════════════════

    var currentTaskId = null;
    var currentFilter = 'all';
    var currentPage = 1;
    var pollTimer = null;
    var selectedFiles = [];
    var batchPreviewUrls = [];
    var batchUploadItems = [];
    var batchValidationPending = false;
    var batchValidationVersion = 0;
    var batchUploadMessage = '';

    // ── 侧边栏导航 ──
    var navSingle = document.getElementById('nav-single-detect');
    var navBatch = document.getElementById('nav-batch-detect');
    var navSpecialSkills = document.getElementById('nav-special-skills');
    var batchView = document.getElementById('batch-view');
    var skillView = document.getElementById('skill-view');
    var skillViewTitle = document.getElementById('skill-view-title');
    var skillDocument = document.getElementById('skill-document');
    var skillTocTitle = document.getElementById('skill-toc-title');
    var skillTocLinks = document.getElementById('skill-toc-links');
    var skillDensityToggle = document.getElementById('skill-density-toggle');
    var skillEditButton = document.getElementById('skill-edit-button');
    var skillExportButton = document.getElementById('skill-export-button');
    var skillEditorPanel = document.getElementById('skill-editor-panel');
    var skillEditorInput = document.getElementById('skill-editor-input');
    var skillEditorSave = document.getElementById('skill-editor-save');
    var skillEditorCancel = document.getElementById('skill-editor-cancel');
    var detectForm = document.getElementById('detect-form');
    var contentHeader = document.querySelector('.main-content .content-header');
    var activeSkillRequest = '';
    var activeSkillKey = '';
    var activeSkillData = null;
    var skillCache = {};
    var skillMenu = {
        common: { title: 'AI 图片鉴定公共模块', slug: 'ai-image-forensics-common.md', isCommon: true },
        fraud: { title: '涉诈鉴别 Skill', slug: 'fraud-ai-image-forensics' },
        rumor: { title: '涉谣鉴别 Skill', slug: 'rumor-ai-image-forensics' },
        porn: { title: '涉黄鉴别 Skill', slug: 'porn-ai-image-forensics' }
    };
    function runViewTransition(applyView) {
        applyView();
    }

    function setSidebarActive(activeId) {
        [navSingle, navBatch, navSpecialSkills].forEach(function (el) {
            if (el) {
                el.classList.remove('active', 'sub-active');
                if (el.id === activeId) {
                    el.classList.add('active');
                }
            }
        });
    }

    function switchToSingleView() {
        runViewTransition(function () {
            if (batchView) batchView.style.display = 'none';
            if (skillView) skillView.style.display = 'none';
            if (detectForm) detectForm.style.display = '';
            if (contentHeader) contentHeader.style.display = '';
            setSidebarActive('nav-single-detect');
            stopPolling();
        });
    }

    function switchToBatchView(view) {
        runViewTransition(function () {
            if (detectForm) detectForm.style.display = 'none';
            if (batchView) batchView.style.display = '';
            if (skillView) skillView.style.display = 'none';
            if (contentHeader) contentHeader.style.display = 'none';
            if (view === 'list' || !view) {
                document.getElementById('task-list-view').style.display = '';
                document.getElementById('task-detail-view').style.display = 'none';
                loadTaskList();
                setSidebarActive('nav-batch-detect');
            } else if (view === 'detail') {
                document.getElementById('task-list-view').style.display = 'none';
                document.getElementById('task-detail-view').style.display = '';
                // 任务详情属于批量鉴定流程，保持“鉴定记录”导航高亮。
                setSidebarActive('nav-batch-detect');
            }
        });
    }

    function escapeSkillHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function formatSkillInline(value) {
        return escapeSkillHtml(value)
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    }

    function isSkillTableDivider(line) {
        return /^\s*\|?\s*:?-{3,}/.test(line || '');
    }

    function skillTableRow(line, tagName) {
        var cells = String(line).trim().replace(/^\||\|$/g, '').split('|');
        return '<tr>' + cells.map(function (cell) {
            return '<' + tagName + '>' + formatSkillInline(cell.trim()) + '</' + tagName + '>';
        }).join('') + '</tr>';
    }

    function renderSkillMarkdown(markdown, idPrefix) {
        var source = String(markdown || '');
        var frontmatter = '';
        var match = source.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/);
        if (match) {
            frontmatter = '---\n' + match[1].trim() + '\n---';
            source = source.slice(match[0].length);
        }
        var html = '';
        var listType = '';
        var tableOpen = false;
        var sectionOpen = false;
        var codeBlockOpen = false;
        var toc = [];
        function closeBlocks() {
            if (listType) { html += '</' + listType + '>'; listType = ''; }
            if (tableOpen) { html += '</tbody></table></div>'; tableOpen = false; }
        }
        source.split(/\r?\n/).forEach(function (rawLine) {
            var line = rawLine.trim();
            if (/^```/.test(line)) {
                closeBlocks();
                html += codeBlockOpen ? '</code></pre>' : '<pre><code>';
                codeBlockOpen = !codeBlockOpen;
                return;
            }
            if (codeBlockOpen) {
                html += escapeSkillHtml(rawLine) + '\n';
                return;
            }
            if (!line) { closeBlocks(); return; }
            if (/^-{3,}$/.test(line)) { closeBlocks(); html += '<hr>'; return; }
            if (/^#\s+/.test(line)) return;
            if (/^##\s+/.test(line)) {
                closeBlocks();
                if (sectionOpen) html += '</section>';
                var sectionTitle = line.replace(/^##\s+/, '');
                var sectionId = (idPrefix || 'skill') + '-section-' + (toc.length + 1);
                toc.push({ id: sectionId, title: sectionTitle });
                html += '<section class="skill-section" id="' + sectionId + '"><h2>' + formatSkillInline(sectionTitle) + '</h2>';
                sectionOpen = true;
                return;
            }
            if (/^###\s+/.test(line)) {
                closeBlocks();
                html += '<h3>' + formatSkillInline(line.replace(/^###\s+/, '')) + '</h3>';
                return;
            }
            if (/^####\s+/.test(line)) {
                closeBlocks();
                html += '<h4>' + formatSkillInline(line.replace(/^####\s+/, '')) + '</h4>';
                return;
            }
            if (/^>\s?/.test(line)) {
                closeBlocks();
                html += '<blockquote>' + formatSkillInline(line.replace(/^>\s?/, '')) + '</blockquote>';
                return;
            }
            if (/^\|/.test(line)) {
                if (isSkillTableDivider(line)) return;
                if (!tableOpen) {
                    html += '<div class="skill-table-wrap"><table class="skill-table"><thead>' + skillTableRow(line, 'th') + '</thead><tbody>';
                    tableOpen = true;
                } else {
                    html += skillTableRow(line, 'td');
                }
                return;
            }
            if (/^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line)) {
                if (tableOpen) { html += '</tbody></table></div>'; tableOpen = false; }
                var expectedList = /^\d+\.\s+/.test(line) ? 'ol' : 'ul';
                if (listType && listType !== expectedList) { html += '</' + listType + '>'; listType = ''; }
                if (!listType) { html += '<' + expectedList + '>'; listType = expectedList; }
                html += '<li>' + formatSkillInline(line.replace(/^([-*]|\d+\.)\s+/, '')) + '</li>';
                return;
            }
            closeBlocks();
            html += '<p>' + formatSkillInline(line) + '</p>';
        });
        closeBlocks();
        if (codeBlockOpen) html += '</code></pre>';
        return { frontmatter: frontmatter, html: html + (sectionOpen ? '</section>' : ''), toc: toc };
    }

    function showSkillView(skillKey, navId) {
        activeSkillRequest = skillKey;
        activeSkillKey = skillKey;
        activeSkillData = null;
        if (detectForm) detectForm.style.display = 'none';
        if (batchView) batchView.style.display = 'none';
        if (skillView) skillView.style.display = '';
        if (contentHeader) contentHeader.style.display = 'none';
        if (skillViewTitle) skillViewTitle.textContent = '正在加载 Skill…';
        if (skillDocument) skillDocument.textContent = '正在读取专项鉴别内容…';
        if (skillDocument) skillDocument.style.display = '';
        if (skillTocLinks) skillTocLinks.innerHTML = '';
        if (skillTocTitle) skillTocTitle.textContent = '鉴别 Skills';
        if (skillEditorPanel) skillEditorPanel.style.display = 'none';
        if (skillEditButton) skillEditButton.style.display = '';
        if (skillDensityToggle) {
            skillDensityToggle.setAttribute('aria-pressed', 'true');
            skillDensityToggle.textContent = '缩略正文';
            skillView.classList.add('is-expanded');
        }
        setSidebarActive(navId);
        stopPolling();

        function drawSkill(data) {
            if (activeSkillRequest !== skillKey) return;
            var rendered = renderSkillMarkdown(data.markdown || '', skillKey);
            activeSkillData = data;
            if (skillEditButton) skillEditButton.style.display = data.editable === false ? 'none' : '';
            if (skillViewTitle) skillViewTitle.textContent = data.title || '图像鉴别 Skill';
            if (skillDocument) skillDocument.innerHTML = rendered.html || '<p>暂无可展示的 Skill 内容。</p>';
            if (skillTocLinks) {
                skillTocLinks.innerHTML = Object.keys(skillMenu).map(function (key) {
                    var menu = skillMenu[key];
                    var selected = key === skillKey ? ' is-selected' : '';
                    var commonClass = menu.isCommon ? ' is-common' : '';
                    var badge = menu.isCommon ? '<span class="skill-library-badge">三个专项共用</span>' : '';
                    var itemHtml = '<button type="button" class="skill-library-item' + selected + commonClass + '" data-skill-key="' + key + '">' + badge + '<strong>' + escapeSkillHtml(menu.title) + '</strong><small>' + escapeSkillHtml(menu.slug) + '</small></button>';
                    if (key !== skillKey) return itemHtml;
                    var chapterButtons = '<div class="skill-toc-tree"><p>└─ ' + escapeSkillHtml(data.file_name || 'SKILL.md') + '</p>' + rendered.toc.map(function (item) {
                        return '<button type="button" class="skill-toc-link" data-skill-target="' + item.id + '"><span>├</span>' + escapeSkillHtml(item.title) + '</button>';
                    }).join('') + '</div>';
                    return itemHtml + chapterButtons;
                }).join('');
                skillTocLinks.querySelectorAll('[data-skill-key]').forEach(function (button) {
                    button.addEventListener('click', function () {
                        showSkillView(button.getAttribute('data-skill-key'), 'nav-special-skills');
                    });
                });
                bindSkillToc();
            }
        }

        if (skillCache[skillKey]) {
            drawSkill(skillCache[skillKey]);
            return;
        }
        fetch('/api/skills/' + encodeURIComponent(skillKey))
            .then(function (response) {
                if (!response.ok) throw new Error('读取失败');
                return response.json();
            })
            .then(function (data) { skillCache[skillKey] = data; drawSkill(data); })
            .catch(function () {
                if (activeSkillRequest !== skillKey) return;
                if (skillViewTitle) skillViewTitle.textContent = 'Skill 暂不可用';
                if (skillDocument) skillDocument.textContent = '未能读取对应的专项鉴别内容，请稍后重试。';
            });
    }

    function bindSkillToc() {
        if (!skillTocLinks) return;
        skillTocLinks.querySelectorAll('[data-skill-target]').forEach(function (button) {
            button.addEventListener('click', function () {
                var target = document.getElementById(button.getAttribute('data-skill-target'));
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });
    }

    function showCombinedSkillsView() {
        var skillKeys = ['fraud', 'rumor', 'porn'];
        activeSkillRequest = 'all-special-skills';
        if (detectForm) detectForm.style.display = 'none';
        if (batchView) batchView.style.display = 'none';
        if (skillView) skillView.style.display = '';
        if (contentHeader) contentHeader.style.display = 'none';
        if (skillViewTitle) skillViewTitle.textContent = '专项图像鉴别 Skills';
        if (skillDocument) skillDocument.textContent = '正在读取专项鉴别内容…';
        if (skillTocLinks) skillTocLinks.innerHTML = '';
        if (skillDensityToggle) {
            skillDensityToggle.setAttribute('aria-pressed', 'true');
            skillDensityToggle.textContent = '缩略正文';
            skillView.classList.add('is-expanded');
        }
        setSidebarActive('nav-special-skills');
        stopPolling();

        function loadSkill(skillKey) {
            if (skillCache[skillKey]) return Promise.resolve(skillCache[skillKey]);
            return fetch('/api/skills/' + encodeURIComponent(skillKey))
                .then(function (response) {
                    if (!response.ok) throw new Error('读取失败');
                    return response.json();
                })
                .then(function (data) { skillCache[skillKey] = data; return data; });
        }

        Promise.all(skillKeys.map(loadSkill)).then(function (items) {
            if (activeSkillRequest !== 'all-special-skills') return;
            var allToc = [];
            var groups = items.map(function (data, index) {
                var key = skillKeys[index];
                var rendered = renderSkillMarkdown(data.markdown || '', key);
                allToc.push({ id: 'skill-' + key, title: data.title, group: true, number: index + 1 });
                rendered.toc.forEach(function (item) {
                    allToc.push({ id: item.id, title: item.title, group: false, number: index + 1 });
                });
                return '<section class="skill-group" id="skill-' + key + '">'
                    + '<header class="skill-group-header"><span>Skill ' + (index + 1) + '</span><h2>' + escapeSkillHtml(data.title) + '</h2><p>' + escapeSkillHtml(data.summary || '') + '</p></header>'
                    + '<div class="skill-group-document">' + (rendered.html || '<p>暂无内容。</p>') + '</div></section>';
            });
            if (skillDocument) skillDocument.innerHTML = groups.join('');
            if (skillTocLinks) {
                skillTocLinks.innerHTML = allToc.map(function (item) {
                    var className = item.group ? 'skill-toc-link skill-toc-group' : 'skill-toc-link skill-toc-child';
                    var prefix = item.group ? item.number : '·';
                    return '<button type="button" class="' + className + '" data-skill-target="' + item.id + '"><span>' + prefix + '</span>' + escapeSkillHtml(item.title) + '</button>';
                }).join('');
                bindSkillToc();
            }
        }).catch(function () {
            if (activeSkillRequest !== 'all-special-skills') return;
            if (skillViewTitle) skillViewTitle.textContent = 'Skill 暂不可用';
            if (skillDocument) skillDocument.textContent = '未能读取全部专项鉴别内容，请稍后重试。';
        });
    }

    if (navSingle) {
        navSingle.addEventListener('click', function (e) {
            e.preventDefault();
            switchToSingleView();
        });
    }
    if (navBatch) {
        navBatch.addEventListener('click', function (e) {
            e.preventDefault();
            switchToBatchView('list');
        });
    }
    // ── 任务创建模态框 ──
    var modalOverlay = document.getElementById('batch-modal-overlay');
    var modalClose = document.getElementById('batch-modal-close');
    var btnCancel = document.getElementById('batch-btn-cancel');
    var btnStart = document.getElementById('batch-btn-start');
    var newTaskBtn = document.getElementById('batch-new-btn');
    var taskNameInput = document.getElementById('batch-task-name');
    var filesInput = document.getElementById('batch-files-input');
    var selectFilesBtn = document.getElementById('batch-select-files');
    var filePlaceholder = document.getElementById('batch-file-placeholder');
    var fileList = document.getElementById('batch-file-list');
    var fileFooter = document.getElementById('batch-file-footer');
    var fileCount = document.getElementById('batch-file-count');
    var selectedFileSource = 'files';
    var uploadResultOverlay = document.getElementById('upload-result-overlay');
    var uploadResultClose = document.getElementById('upload-result-close');
    var uploadResultConfirm = document.getElementById('upload-result-confirm');
    var uploadSuccessCount = document.getElementById('upload-success-count');
    var uploadFailedCount = document.getElementById('upload-failed-count');
    var uploadResultDetails = document.getElementById('upload-result-details');
    var uploadFailedDetailCount = document.getElementById('upload-failed-detail-count');
    var uploadResultList = document.getElementById('upload-result-list');

    // 批量任务弹窗原本位于隐藏的“鉴定记录”容器中；提升到页面顶层，
    // 使用户在主上传页选择多张图片时也能立即看到任务命名与缩略图预览。
    if (modalOverlay && modalOverlay.parentNode !== document.body) {
        document.body.appendChild(modalOverlay);
    }

    function hideUploadOutcomeModal() {
        if (uploadResultOverlay) uploadResultOverlay.style.display = 'none';
    }

    function showUploadOutcomeModal(successCount, failedFiles) {
        if (!uploadResultOverlay) return;
        var failures = Array.isArray(failedFiles) ? failedFiles : [];
        if (uploadSuccessCount) uploadSuccessCount.textContent = String(Math.max(0, successCount || 0));
        if (uploadFailedCount) uploadFailedCount.textContent = String(failures.length);
        if (uploadFailedDetailCount) uploadFailedDetailCount.textContent = String(failures.length);
        if (uploadResultDetails) uploadResultDetails.hidden = failures.length === 0;
        if (uploadResultList) {
            uploadResultList.innerHTML = failures.map(function (item) {
                var filename = item && (item.filename || item.name) ? (item.filename || item.name) : '未知文件';
                var reason = item && item.reason ? item.reason : '图片未通过上传校验。';
                return '<div class="upload-result-item"><strong>' + escapeHtml(filename) + '</strong><p>' + escapeHtml(reason) + '</p></div>';
            }).join('');
        }
        uploadResultOverlay.style.display = 'flex';
    }

    if (uploadResultClose) uploadResultClose.addEventListener('click', hideUploadOutcomeModal);
    if (uploadResultConfirm) uploadResultConfirm.addEventListener('click', hideUploadOutcomeModal);
    if (uploadResultOverlay) {
        uploadResultOverlay.addEventListener('click', function (event) {
            if (event.target === uploadResultOverlay) hideUploadOutcomeModal();
        });
    }

    function getDefaultTaskName() {
        var now = new Date();
        function pad(value) { return String(value).padStart(2, '0'); }
        return '图像鉴定 ' + now.getFullYear() + '-' + pad(now.getMonth() + 1) + '-' + pad(now.getDate()) + ' ' + pad(now.getHours()) + ':' + pad(now.getMinutes()) + ':' + pad(now.getSeconds());
    }

    function syncBatchOptionsFromSingle() {
        var singleModels = getSelectedModels(document.querySelector('#detect-form .upload-model-grid'));
        var batchModels = document.getElementById('batch-models');
        if (batchModels) {
            batchModels.querySelectorAll('.upload-model-input').forEach(function (input) {
                input.checked = singleModels.indexOf(input.value) !== -1;
            });
            syncModelCards(batchModels);
        }
        var singleScene = getSelectedCrimeScene(document.querySelector('#detect-form .crime-scene-grid'));
        var batchScenes = document.getElementById('batch-crime-scenes');
        if (batchScenes) {
            batchScenes.querySelectorAll('.crime-scene-input').forEach(function (input) {
                input.checked = input.value === singleScene;
            });
            syncCrimeSceneCards(batchScenes);
        }
    }

    function showTaskCreateModal(initialFiles, source) {
        if (modalOverlay) modalOverlay.style.display = 'flex';
        if (taskNameInput) taskNameInput.value = getDefaultTaskName();
        if (btnStart) btnStart.disabled = true;
        selectedFiles = [];
        batchUploadItems = [];
        batchValidationPending = false;
        batchUploadMessage = '';
        batchValidationVersion += 1;
        resetFileSelector();
        if (initialFiles && initialFiles.length) {
            setBatchSelectedFiles(initialFiles, source || 'files');
        } else {
            updateBatchStartBtn();
        }
        if (taskNameInput) taskNameInput.focus();
    }

    function hideTaskCreateModal() {
        if (modalOverlay) modalOverlay.style.display = 'none';
        clearBatchPreviewUrls();
    }

    function resetFileSelector() {
        clearBatchPreviewUrls();
        if (filePlaceholder) filePlaceholder.style.display = '';
        if (fileList) fileList.style.display = 'none';
        if (fileFooter) fileFooter.style.display = 'none';
        if (fileCount) fileCount.textContent = '已选择 0 张图片';
        if (filesInput) filesInput.value = '';
    }

    function clearBatchPreviewUrls() {
        batchPreviewUrls.forEach(function (url) { URL.revokeObjectURL(url); });
        batchPreviewUrls = [];
    }

    function renderBatchSelectedFiles() {
        var imageItems = batchUploadItems;
        if (imageItems.length === 0) {
            resetFileSelector();
            updateBatchStartBtn();
            return;
        }
        if (filePlaceholder) filePlaceholder.style.display = 'none';
        if (fileList) {
            fileList.style.display = '';
            clearBatchPreviewUrls();
            var validCount = imageItems.filter(function (item) { return item.valid === true; }).length;
            var invalidCount = imageItems.filter(function (item) { return item.valid === false; }).length;
            var title = '已选择 <strong>' + imageItems.length + '</strong> 张图片';
            if (batchValidationPending) title += '<span class="batch-selection-checking">正在校验图片…</span>';
            else title += '<span class="batch-selection-valid">可上传 ' + validCount + ' 张</span>' + (invalidCount ? '<span class="batch-selection-invalid">不可上传 ' + invalidCount + ' 张</span>' : '');
            if (batchUploadMessage) title += '<p class="batch-selection-message">' + escapeHtml(batchUploadMessage) + '</p>';
            var html = '<div class="batch-selection-title">' + title + '</div><div class="batch-preview-grid">';
            var showCount = Math.min(imageItems.length, 20);
            for (var j = 0; j < showCount; j++) {
                var item = imageItems[j];
                var previewUrl = URL.createObjectURL(item.file);
                batchPreviewUrls.push(previewUrl);
                var stateClass = item.valid === true ? 'is-valid' : (item.valid === false ? 'is-invalid' : 'is-checking');
                var stateText = item.valid === true ? '可上传' : (item.valid === false ? item.reason : '正在校验…');
                html += '<figure class="batch-preview-item ' + stateClass + '"><button type="button" class="batch-preview-remove" data-remove-batch-index="' + j + '" aria-label="删除 ' + escapeHtml(item.file.name) + '">×</button><img src="' + previewUrl + '" alt="' + escapeHtml(item.file.name) + '"><figcaption>' + escapeHtml(item.file.name) + '</figcaption><span class="batch-preview-status">' + escapeHtml(stateText) + '</span></figure>';
            }
            if (imageItems.length > 20) html += '<div class="batch-preview-more">另有 ' + (imageItems.length - 20) + ' 张图片</div>';
            fileList.innerHTML = html + '</div>';
        }
        if (fileFooter) fileFooter.style.display = '';
        if (fileCount) fileCount.textContent = '已选择 ' + imageItems.length + ' 张图片，其中可上传 ' + selectedFiles.length + ' 张';
        updateBatchStartBtn();
    }

    function discardBatchFiles(skippedFiles) {
        if (!Array.isArray(skippedFiles) || skippedFiles.length === 0) return 0;
        var skippedIndexes = {};
        skippedFiles.forEach(function (item) {
            if (item && typeof item.index === 'number') skippedIndexes[item.index] = true;
        });
        var before = selectedFiles.length;
        skippedFiles.forEach(function (item) {
            var skippedFile = typeof item.index === 'number' ? selectedFiles[item.index] : null;
            batchUploadItems.forEach(function (uploadItem) {
                if (skippedFile && uploadItem.file === skippedFile) {
                    uploadItem.valid = false;
                    uploadItem.reason = item.reason || '图片未通过检测质量校验。';
                }
            });
        });
        selectedFiles = selectedFiles.filter(function (_file, index) { return !skippedIndexes[index]; });
        renderBatchSelectedFiles();
        return before - selectedFiles.length;
    }

    if (newTaskBtn) {
        newTaskBtn.addEventListener('click', showTaskCreateModal);
    }
    if (modalClose) {
        modalClose.addEventListener('click', hideTaskCreateModal);
    }
    if (btnCancel) {
        btnCancel.addEventListener('click', hideTaskCreateModal);
    }
    // 点击遮罩关闭
    if (modalOverlay) {
        modalOverlay.addEventListener('click', function (e) {
            if (e.target === modalOverlay) hideTaskCreateModal();
        });
    }

    function openBatchFilePicker() {
        selectedFileSource = 'files';
        if (filesInput) filesInput.click();
    }
    if (navSpecialSkills) {
        navSpecialSkills.addEventListener('click', function (e) {
            e.preventDefault();
            showSkillView(activeSkillKey || 'fraud', 'nav-special-skills');
        });
    }
    if (skillDensityToggle) {
        skillDensityToggle.addEventListener('click', function () {
            var expanded = skillView.classList.toggle('is-expanded');
            skillDensityToggle.setAttribute('aria-pressed', expanded ? 'true' : 'false');
            skillDensityToggle.textContent = expanded ? '缩略正文' : '展开完整正文';
        });
    }
    if (skillEditButton) {
        skillEditButton.addEventListener('click', function () {
            if (!activeSkillData || !skillEditorInput) return;
            skillEditorInput.value = activeSkillData.markdown || '';
            if (skillDocument) skillDocument.style.display = 'none';
            if (skillEditorPanel) skillEditorPanel.style.display = '';
            skillEditorInput.focus();
        });
    }
    if (skillEditorCancel) {
        skillEditorCancel.addEventListener('click', function () {
            if (skillEditorPanel) skillEditorPanel.style.display = 'none';
            if (skillDocument) skillDocument.style.display = '';
        });
    }
    if (skillEditorSave) {
        skillEditorSave.addEventListener('click', function () {
            if (!activeSkillKey || !skillEditorInput) return;
            skillEditorSave.disabled = true;
            skillEditorSave.textContent = '正在保存…';
            fetch('/api/skills/' + encodeURIComponent(activeSkillKey), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ markdown: skillEditorInput.value })
            }).then(function (response) {
                return response.json().then(function (data) {
                    if (!response.ok) throw new Error(data.error || '保存失败');
                    return data;
                });
            }).then(function (data) {
                skillCache[activeSkillKey] = Object.assign({}, activeSkillData, { markdown: data.markdown });
                if (skillEditorPanel) skillEditorPanel.style.display = 'none';
                if (skillDocument) skillDocument.style.display = '';
                showSkillView(activeSkillKey, 'nav-special-skills');
            }).catch(function (error) {
                window.alert(error.message || '保存 Skill 失败，请稍后重试。');
            }).finally(function () {
                skillEditorSave.disabled = false;
                skillEditorSave.textContent = '保存更改';
            });
        });
    }
    if (skillExportButton) {
        skillExportButton.addEventListener('click', function () {
            if (!activeSkillData || !activeSkillData.markdown) return;
            var blob = new Blob([activeSkillData.markdown], { type: 'text/markdown;charset=utf-8' });
            var downloadUrl = URL.createObjectURL(blob);
            var link = document.createElement('a');
            link.href = downloadUrl;
            link.download = (activeSkillData.skill_name || activeSkillKey || 'skill') + '.md';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(downloadUrl);
        });
    }

    // 多图选择
    if (selectFilesBtn) {
        selectFilesBtn.addEventListener('click', function () { openBatchFilePicker(); });
    }
    if (fileList) {
        fileList.addEventListener('click', function (event) {
            var removeBtn = event.target.closest('.batch-preview-remove');
            if (!removeBtn) return;
            event.preventDefault();
            var index = Number(removeBtn.getAttribute('data-remove-batch-index'));
            if (!Number.isInteger(index) || index < 0 || index >= batchUploadItems.length) return;
            batchUploadItems.splice(index, 1);
            selectedFiles = batchUploadItems.filter(function (item) { return item.valid === true; }).map(function (item) { return item.file; });
            batchUploadMessage = '';
            renderBatchSelectedFiles();
        });
    }

    function validateBatchUploadItem(item) {
        var file = item.file;
        if (!/\.(jpe?g|png|bmp|webp)$/i.test(file.name || '')) {
            return Promise.resolve({ valid: false, reason: '仅支持 JPG、JPEG、PNG、BMP 或 WebP 图片。' });
        }
        if (file.size < MIN_IMAGE_FILE_BYTES) {
            return Promise.resolve({ valid: false, reason: '图片文件大小不能小于 30 KB。' });
        }
        if (file.size > MAX_IMAGE_FILE_BYTES) {
            return Promise.resolve({ valid: false, reason: '图片文件大小不能超过 20 MB。' });
        }
        return new Promise(function (resolve) {
            var image = new Image();
            var imageUrl = URL.createObjectURL(file);
            image.onload = function () {
                URL.revokeObjectURL(imageUrl);
                if (Math.min(image.naturalWidth, image.naturalHeight) < MIN_IMAGE_SHORT_EDGE) {
                    resolve({ valid: false, reason: '图片最小尺寸不小于 300px（当前 ' + image.naturalWidth + ' × ' + image.naturalHeight + '）。' });
                } else {
                    resolve({ valid: true, reason: '' });
                }
            };
            image.onerror = function () {
                URL.revokeObjectURL(imageUrl);
                resolve({ valid: false, reason: '无法读取图片，请上传完整的图片文件。' });
            };
            image.src = imageUrl;
        });
    }

    function setBatchSelectedFiles(files, source) {
        if (!files || files.length === 0) return;
        var validationVersion = ++batchValidationVersion;
        batchUploadItems = Array.prototype.slice.call(files).map(function (file) {
            return { file: file, valid: null, reason: '' };
        });
        selectedFiles = [];
        selectedFileSource = source;
        batchUploadMessage = '';
        batchValidationPending = true;
        renderBatchSelectedFiles();
        Promise.all(batchUploadItems.map(function (item) {
            return validateBatchUploadItem(item).then(function (result) {
                item.valid = result.valid;
                item.reason = result.reason;
            });
        })).then(function () {
            if (validationVersion !== batchValidationVersion) return;
            selectedFiles = batchUploadItems.filter(function (item) { return item.valid === true; }).map(function (item) { return item.file; });
            batchValidationPending = false;
            renderBatchSelectedFiles();
        });
    }

    if (filesInput) {
        filesInput.addEventListener('change', function () { setBatchSelectedFiles(filesInput.files, 'files'); });
    }

    function updateBatchStartBtn() {
        var name = taskNameInput ? taskNameInput.value.trim() : '';
        if (btnStart) {
            btnStart.disabled = !(name && selectedFiles.length > 0 && !batchValidationPending);
        }
    }

    if (taskNameInput) {
        taskNameInput.addEventListener('input', updateBatchStartBtn);
    }

    // ── 提交批量任务 ──
    if (btnStart) {
        btnStart.addEventListener('click', function () {
            var name = taskNameInput ? taskNameInput.value.trim() : '';
            if (!name || selectedFiles.length === 0) return;

            btnStart.disabled = true;
            btnStart.textContent = '创建中...';

            var formData = new FormData();
            formData.append('task_name', name);
            var selectedModels = getSelectedModels(document.getElementById('batch-models'));
            if (selectedModels.length === 0) {
                alert('请至少选择一项检测模型');
                btnStart.disabled = false;
                btnStart.textContent = '开始检测';
                return;
            }
            var selectedCrimeScene = getSelectedCrimeScene(document.getElementById('batch-crime-scenes'));
            selectedModels.forEach(function (model) { formData.append('models', model); });
            if (selectedCrimeScene) formData.append('crime_scene', selectedCrimeScene);
            for (var k = 0; k < selectedFiles.length; k++) {
                formData.append('files', selectedFiles[k]);
            }

            fetch('/api/batch/detect', {
                method: 'POST',
                body: formData
            })
            .then(function (res) { return res.json(); })
            .then(function (res) {
                if (res.code === 200) {
                    var skippedFiles = res.data && res.data.skipped_files;
                    var removed = discardBatchFiles(skippedFiles);
                    var successCount = res.data && typeof res.data.total_count === 'number' ? res.data.total_count : Math.max(0, selectedFiles.length);
                    if (removed > 0) {
                        batchUploadItems = batchUploadItems.filter(function (item) { return item.valid === false; });
                        selectedFiles = [];
                        batchUploadMessage = '鉴定任务已创建，成功加入 ' + successCount + ' 张图片；以下图片无法上传，原因已标注。';
                        renderBatchSelectedFiles();
                    } else {
                        hideTaskCreateModal();
                        switchToBatchView('list');
                    }
                } else {
                    var failedFiles = res.skipped_files || [];
                    var removed = discardBatchFiles(failedFiles);
                    if (removed > 0) {
                        batchUploadMessage = '以下图片无法上传，原因已标注。请删除或重新选择后再提交。';
                        renderBatchSelectedFiles();
                    } else {
                        batchUploadMessage = res.msg || '创建任务失败，请稍后重试。';
                        renderBatchSelectedFiles();
                    }
                }
            })
            .catch(function (err) {
                batchUploadMessage = '请求异常：' + err.message;
                renderBatchSelectedFiles();
            })
            .finally(function () {
                btnStart.textContent = '开始检测';
                updateBatchStartBtn();
            });
        });
    }

    // ── 任务列表（表格版）──
    var taskSearchVal = '';
    var taskStatusVal = '';
    var taskPage = 1;
    var taskPageSize = 20;

    function loadTaskList() {
        stopPolling();
        var tbody = document.getElementById('task-table-body');
        var empty = document.getElementById('task-table-empty');
        var countInfo = document.getElementById('task-count-info');
        if (!tbody) return;

        // 读取搜索和筛选值
        var searchInput = document.getElementById('task-search-input');
        var statusSelect = document.getElementById('task-status-select');
        taskSearchVal = searchInput ? searchInput.value.trim() : '';
        taskStatusVal = statusSelect ? statusSelect.value : '';

        var url = '/api/batch/tasks?' +
            'search=' + encodeURIComponent(taskSearchVal) +
            '&status=' + encodeURIComponent(taskStatusVal) +
            '&page=' + taskPage +
            '&page_size=' + taskPageSize;

        fetch(url)
        .then(function (res) { return res.json(); })
        .then(function (res) {
            if (res.code !== 200) return;
            var data = res.data || {};
            var tasks = data.tasks || [];
            var total = data.total || 0;
            var totalPages = data.total_pages || 1;

            // 更新计数信息
            if (countInfo) countInfo.textContent = '共找到 ' + total + ' 条鉴定记录';

            if (tasks.length === 0) {
                tbody.innerHTML = '';
                if (empty) empty.style.display = '';
                renderTaskPagination(total, totalPages);
                return;
            }

            if (empty) empty.style.display = 'none';

            // 渲染表格行
            var html = '';
            var hasRunning = false;
            tasks.forEach(function (task) {
                html += renderTaskRow(task);
                if (task.status === 'running') hasRunning = true;
            });
            tbody.innerHTML = html;

            // 绑定操作按钮
            tbody.querySelectorAll('.action-view-btn').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var tid = this.getAttribute('data-task-id');
                    if (tid) loadTaskDetail(tid);
                });
            });
            tbody.querySelectorAll('.action-delete-btn').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var tid = this.getAttribute('data-task-id');
                    var tname = this.getAttribute('data-task-name');
                    if (tid) deleteTask(tid, tname);
                });
            });

            // 更新分页
            renderTaskPagination(total, totalPages);

            // 如果有运行中的任务，启动轮询
            if (hasRunning) startPolling();
        })
        .catch(function () {});
    }

    function renderTaskRow(task) {
        var pct = task.total_count > 0 ? Math.round(task.completed_count / task.total_count * 100) : 0;
        var statusClass = task.status;
        var statusIcon = task.status === 'running' ? '🔄' : (task.status === 'completed' ? '✅' : '❌');
        var statusLabel = task.status === 'running' ? '运行中' : (task.status === 'completed' ? '已完成' : '失败');

        return '<tr>' +
            '<td class="col-check"><input type="checkbox" class="task-checkbox" data-task-id="' + task.task_id + '"></td>' +
            '<td class="col-name"><strong>' + escapeHtml(task.task_name) + '</strong></td>' +
            '<td class="col-status"><span class="status-badge ' + statusClass + '">' + statusIcon + ' ' + statusLabel + '</span></td>' +
            '<td class="col-progress">' +
                '<div class="tc-progress">' +
                    '<div class="tc-progress-bar"><div class="tc-progress-fill" style="width:' + pct + '%"></div></div>' +
                    '<span class="tc-progress-text">' + task.completed_count + '/' + task.total_count + '</span>' +
                '</div>' +
            '</td>' +
            '<td class="col-stats">' +
                '<span style="color:#dc2626;">🔴' + task.ai_count + '</span> ' +
                '<span style="color:#d97706;">🟡' + task.suspected_count + '</span> ' +
                '<span style="color:#16a34a;">🟢' + task.real_count + '</span>' +
                (task.error_count > 0 ? ' <span style="color:#999;">⚠️' + task.error_count + '</span>' : '') +
            '</td>' +
            '<td class="col-time" style="font-size:12px;color:#999;">' + (task.created_at || '') + '</td>' +
            '<td class="col-actions"><div class="tc-actions">' +
                '<button class="action-view-btn" data-task-id="' + task.task_id + '">查看</button>' +
                '<button class="action-delete-btn" data-task-id="' + task.task_id + '" data-task-name="' + escapeHtml(task.task_name) + '">删除</button>' +
            '</div></td>' +
        '</tr>';
    }

    // ── 删除任务 ──
    function deleteTask(taskId, taskName) {
        if (!confirm('确定要删除任务「' + taskName + '」吗？此操作不可撤销。')) return;
        fetch('/api/batch/task/' + taskId, { method: 'DELETE' })
        .then(function (res) { return res.json(); })
        .then(function (res) {
            if (res.code === 200) {
                loadTaskList();
            } else {
                alert('删除失败：' + (res.msg || '未知错误'));
            }
        })
        .catch(function (err) {
            alert('请求异常：' + err.message);
        });
    }

    // ── 任务列表分页 ──
    function renderTaskPagination(total, totalPages) {
        var pag = document.getElementById('task-pagination');
        var pageNum = document.getElementById('page-num-text');
        var prevBtn = document.getElementById('page-prev');
        var nextBtn = document.getElementById('page-next');

        if (!pag) return;

        pag.style.display = '';

        if (pageNum) pageNum.textContent = taskPage + ' / ' + totalPages;
        if (prevBtn) {
            prevBtn.disabled = taskPage <= 1;
            prevBtn.onclick = function () {
                if (taskPage > 1) { taskPage--; loadTaskList(); }
            };
        }
        if (nextBtn) {
            nextBtn.disabled = taskPage >= totalPages;
            nextBtn.onclick = function () {
                if (taskPage < totalPages) { taskPage++; loadTaskList(); }
            };
        }
    }

    // ── 搜索/筛选事件绑定 ──
    var searchInput = document.getElementById('task-search-input');
    var statusSelect = document.getElementById('task-status-select');
    var searchBtn = document.getElementById('task-search-btn');
    var resetBtn = document.getElementById('task-reset-btn');
    var refreshBtn = document.getElementById('task-refresh-btn');
    var selectAll = document.getElementById('select-all');
    var batchDeleteSelected = document.getElementById('batch-delete-selected');

    function doTaskSearch() {
        taskPage = 1;
        loadTaskList();
    }

    if (searchBtn) {
        searchBtn.addEventListener('click', doTaskSearch);
    }
    if (searchInput) {
        searchInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') doTaskSearch();
        });
    }
    if (statusSelect) {
        statusSelect.addEventListener('change', doTaskSearch);
    }
    if (resetBtn) {
        resetBtn.addEventListener('click', function () {
            if (searchInput) searchInput.value = '';
            if (statusSelect) statusSelect.value = '';
            taskPage = 1;
            loadTaskList();
        });
    }
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function () {
            loadTaskList();
        });
    }

    // ── 每页条数切换 ──
    var pageSizeSelect = document.getElementById('page-size-select');
    if (pageSizeSelect) {
        pageSizeSelect.addEventListener('change', function () {
            taskPageSize = parseInt(this.value);
            taskPage = 1;
            loadTaskList();
        });
    }

    // ── 全选 / 批量删除 ──
    if (selectAll) {
        selectAll.addEventListener('change', function () {
            var checked = this.checked;
            document.querySelectorAll('.task-checkbox').forEach(function (cb) {
                cb.checked = checked;
            });
            updateBatchDeleteBtn();
        });
    }

    // 委托事件：点击表格中的复选框时更新批量删除按钮状态
    document.getElementById('task-table') && document.getElementById('task-table').addEventListener('change', function (e) {
        if (e.target && e.target.classList.contains('task-checkbox')) {
            updateBatchDeleteBtn();
        }
    });

    function updateBatchDeleteBtn() {
        var checked = document.querySelectorAll('.task-checkbox:checked');
        if (batchDeleteSelected) {
            batchDeleteSelected.disabled = checked.length === 0;
            batchDeleteSelected.textContent = '🗑️ 批量删除' + (checked.length > 0 ? ' (' + checked.length + ')' : '');
        }
    }

    if (batchDeleteSelected) {
        batchDeleteSelected.addEventListener('click', function () {
            var checked = document.querySelectorAll('.task-checkbox:checked');
            if (checked.length === 0) return;
            if (!confirm('确定要删除选中的 ' + checked.length + ' 个任务吗？此操作不可撤销。')) return;

            var ids = [];
            checked.forEach(function (cb) { ids.push(cb.getAttribute('data-task-id')); });

            // 逐个删除（后端尚未提供批量删除接口）
            var deleted = 0;
            var errors = 0;
            function delNext(idx) {
                if (idx >= ids.length) {
                    var msg = '已删除 ' + deleted + ' 个任务';
                    if (errors > 0) msg += '，' + errors + ' 个失败';
                    alert(msg);
                    loadTaskList();
                    return;
                }
                fetch('/api/batch/task/' + ids[idx], { method: 'DELETE' })
                .then(function (r) { return r.json(); })
                .then(function (r) {
                    if (r.code === 200) deleted++;
                    else errors++;
                    delNext(idx + 1);
                })
                .catch(function () { errors++; delNext(idx + 1); });
            }
            delNext(0);
        });
    }

    // ── 轮询 ──
    function startPolling() {
        stopPolling();
        pollTimer = setInterval(function () {
            loadTaskList();
        }, 3000);
    }

    function stopPolling() {
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }

    // ── 任务详情 ──
    var backBtn = document.getElementById('batch-back-btn');

    if (backBtn) {
        backBtn.addEventListener('click', function () {
            switchToBatchView('list');
        });
    }

    function loadTaskDetail(taskId) {
        currentTaskId = taskId;
        currentFilter = 'all';
        currentPage = 1;
        switchToBatchView('detail');

        // 获取任务概要
        fetch('/api/batch/task/' + taskId)
        .then(function (res) { return res.json(); })
        .then(function (res) {
            if (res.code === 200 && res.data) {
                var task = res.data;
                document.getElementById('task-detail-name').textContent = task.task_name;
                var stats = '<span class="task-stat task-stat-total"><b>' + task.total_count + '</b> 张图片</span>' +
                    '<span class="task-stat task-stat-ai"><i></i>AI 伪造 <b>' + task.ai_count + '</b></span>' +
                    '<span class="task-stat task-stat-suspected"><i></i>疑似 AI <b>' + task.suspected_count + '</b></span>' +
                    '<span class="task-stat task-stat-real"><i></i>真实图片 <b>' + task.real_count + '</b></span>';
                if (task.error_count > 0) stats += '<span class="task-stat task-stat-error"><i></i>失败 <b>' + task.error_count + '</b></span>';
                document.getElementById('task-detail-stats').innerHTML = stats;
            }
        })
        .catch(function () {});

        loadTaskImages(taskId, 'all', 1);
    }

    function loadTaskImages(taskId, filter, page) {
        var url = '/api/batch/task/' + taskId + '/images?page=' + page + '&page_size=20';
        if (filter && filter !== 'all') url += '&filter=' + filter;

        fetch(url)
        .then(function (res) { return res.json(); })
        .then(function (res) {
            if (res.code === 200 && res.data) {
                renderImageGrid(res.data);
                renderPagination(res.data);
            }
        })
        .catch(function () {});
    }

    function renderImageGrid(data) {
        var grid = document.getElementById('image-grid');
        if (!grid) return;

        var images = data.images || [];
        if (images.length === 0) {
            grid.innerHTML = '<p class="batch-empty">暂无图片</p>';
            return;
        }

        var html = '';
        images.forEach(function (img) {
            html += '<div class="image-card" data-image-id="' + img.image_id + '" data-task-id="' + currentTaskId + '">' +
                '<img class="image-card-thumb" src="/uploads/' + img.saved_path.split('\\').pop().split('/').pop() + '" alt="' + escapeHtml(img.filename) + '" loading="lazy">' +
                '<div class="image-card-body">' +
                    '<div class="image-card-name" title="' + escapeHtml(img.filename) + '">' + escapeHtml(img.filename) + '</div>' +
                    '<div style="display:flex;align-items:center;gap:6px;">' +
                        '<span class="image-card-badge ' + (img.label || 'unknown') + '">' +
                            (img.label === 'ai_generated' ? '🔴' : (img.label === 'suspected_ai' ? '🟡' : '🟢')) + ' ' +
                            (img.label_text || '未知') +
                        '</span>' +
                        '<span class="image-card-conf">' +
                            (img.confidence ? (img.confidence * 100).toFixed(1) + '%' : '--') +
                        '</span>' +
                    '</div>' +
                '</div>' +
            '</div>';
        });
        grid.innerHTML = html;

        // 绑定点击事件 → 查看图片详情
        grid.querySelectorAll('.image-card').forEach(function (card) {
            card.addEventListener('click', function () {
                var tid = this.getAttribute('data-task-id');
                var iid = this.getAttribute('data-image-id');
                if (tid && iid) showImageDetail(tid, iid);
            });
        });
    }

    function renderPagination(data) {
        var pag = document.getElementById('pagination');
        if (!pag) return;

        if (data.total_pages <= 1) {
            pag.style.display = 'none';
            return;
        }
        pag.style.display = '';

        var html = '';
        var p = data.page;
        var tp = data.total_pages;

        html += '<button class="page-btn" data-page="' + (p - 1) + '"' + (p <= 1 ? ' disabled' : '') + '>‹ 上一页</button>';
        html += '<span class="page-info">第 ' + p + '/' + tp + ' 页</span>';
        html += '<button class="page-btn" data-page="' + (p + 1) + '"' + (p >= tp ? ' disabled' : '') + '>下一页 ›</button>';

        pag.innerHTML = html;

        pag.querySelectorAll('.page-btn:not([disabled])').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var goPage = parseInt(this.getAttribute('data-page'));
                if (goPage && currentTaskId) {
                    currentPage = goPage;
                    loadTaskImages(currentTaskId, currentFilter, goPage);
                }
            });
        });
    }

    // ── 筛选 ──
    document.querySelectorAll('.batch-filter-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.batch-filter-btn').forEach(function (b) { b.classList.remove('active'); });
            this.classList.add('active');
            currentFilter = this.getAttribute('data-filter') || 'all';
            currentPage = 1;
            if (currentTaskId) {
                loadTaskImages(currentTaskId, currentFilter, 1);
            }
        });
    });

    // ── 图片详情弹窗 ──
    var imageDetailOverlay = document.getElementById('image-detail-overlay');
    var imageDetailClose = document.getElementById('image-detail-close');
    var imageDetailBody = document.getElementById('image-detail-body');

    if (imageDetailClose) {
        imageDetailClose.addEventListener('click', function () {
            if (imageDetailOverlay) imageDetailOverlay.style.display = 'none';
        });
    }
    if (imageDetailOverlay) {
        imageDetailOverlay.addEventListener('click', function (e) {
            if (e.target === imageDetailOverlay) imageDetailOverlay.style.display = 'none';
        });
    }

    function showImageDetail(taskId, imageId) {
        fetch('/api/batch/task/' + taskId + '/image/' + imageId)
        .then(function (res) { return res.json(); })
        .then(function (res) {
            if (res.code === 200 && res.data) {
                var img = res.data;
                document.getElementById('image-detail-title').textContent = '🔍 ' + img.filename;

                // 构建分析面板内容
                var html = '';

                // 预览图 + 基本信息
                html += '<div style="display:flex;gap:16px;margin-bottom:20px;">';
                html += '<img src="/uploads/' + img.saved_path.split('\\').pop().split('/').pop() + '" style="max-width:280px;max-height:200px;border-radius:8px;object-fit:contain;box-shadow:0 2px 8px rgba(0,0,0,0.08);">';
                html += '<div style="flex:1;">';
                html += '<p style="margin:4px 0;"><strong>文件名：</strong>' + escapeHtml(img.filename) + '</p>';
                html += '<p style="margin:4px 0;"><strong>判定：</strong><span class="image-card-badge ' + (img.label || '') + '" style="font-size:14px;">' + (img.label_text || '未知') + '</span></p>';
                html += '<p style="margin:4px 0;"><strong>置信度：</strong>' + (img.confidence ? (img.confidence * 100).toFixed(2) + '%' : '--') + '</p>';
                html += '</div></div>';

                // 如果有水印结果
                var batchWatermark = img.watermark_result || {};
                var batchC2pa = (batchWatermark.c2pa_verification && batchWatermark.c2pa_verification.present) ||
                    (batchWatermark.metadata && batchWatermark.metadata.c2pa && batchWatermark.metadata.c2pa.present);
                if (batchWatermark.detected && !batchC2pa) {
                    html += '<div class="watermark-notice" style="margin-bottom:16px;">' +
                        '<div class="wn-header"><span class="wn-icon">🏷️</span>' +
                        '<span class="wn-title">检测到 AI 生成水印标识</span></div>' +
                        '<div class="wn-body"><div class="wn-row">' +
                        '<span class="wn-label">来源平台</span>' +
                        '<span class="wn-value">' + (batchWatermark.source || '未知') + '</span></div></div></div>';
                }

                // AI 图片内容识别 + 知识库风险研判（只展示系统已完成的研判结果）
                var content = img.content_analysis;
                if (img.label === 'ai_generated' && content && content.status !== 'skipped') {
                    var riskLevel = content.risk_level || '未知';
                    var riskColor = riskLevel === '高' ? '#dc2626' : (riskLevel === '中' ? '#d97706' : (riskLevel === '低' ? '#16a34a' : '#6b7280'));
                    var chips = function (values, risk) {
                        values = values || [];
                        if (!values.length) return '<span class="content-risk-chip">未识别</span>';
                        return values.map(function (item) {
                            var label = typeof item === 'string' ? item : item.tag;
                            var itemLevel = typeof item === 'string' ? '' : item.level;
                            var cls = risk ? (itemLevel === '高' ? ' risk-high' : (itemLevel === '中' ? ' risk-medium' : ' risk-low')) : '';
                            return '<span class="content-risk-chip' + cls + '">' + escapeHtml(label || '未命名') + '</span>';
                        }).join('');
                    };
                    html += '<div class="content-risk-section" style="margin-bottom:16px;">' +
                        '<div class="content-risk-header"><div><div class="content-risk-title">图片内容识别与犯罪风险研判</div>' +
                        '<div class="content-risk-subtitle">视觉内容、OCR 线索经知识库规则匹配，仅作风险提示，不构成事实认定</div></div>' +
                        '<span class="content-risk-level" style="background:' + riskColor + ';">' + escapeHtml(riskLevel) + '风险</span></div>' +
                        ((content.crime_scene_label || (content.forensic_context || {}).crime_scene_label) ? '<div class="content-risk-scope">犯罪场景：' + escapeHtml(content.crime_scene_label || (content.forensic_context || {}).crime_scene_label) + '</div>' : '') +
                        '<div class="content-risk-description">' + escapeHtml(content.scene_description || content.detail || '内容识别服务暂不可用。') + '</div>' +
                        '<div class="content-risk-row"><span>内容标签</span><div class="content-risk-chips">' + chips(content.content_tags, false) + '</div></div>' +
                        '<div class="content-risk-row"><span>风险标签</span><div class="content-risk-chips">' + chips(content.risk_tags, true) + '</div></div>' +
                        '<div class="content-risk-row"><span>知识库匹配</span><div class="content-risk-chips">' + chips(content.matched_knowledge, false) + '</div></div>' +
                        '<div class="content-risk-evidence"><strong>识别依据：</strong>' + escapeHtml((content.evidence || []).join('；') || '暂无可展示的识别依据。') + '</div>' +
                        '<div class="content-risk-checks"><strong>建议核查：</strong>' + escapeHtml((content.recommended_checks || []).join('；') || '暂无建议核查项。') + '</div>' +
                        ((content.ocr_text || []).length ? '<div class="content-risk-ocr">OCR 识别文本：' + escapeHtml(content.ocr_text.join(' | ')) + '</div>' : '') +
                        '</div>';
                }

                // ── AI全图生成分析（优先 combined_result，降级到 npr_result）──
                var aiScore, aiVerdict;
                if (img.combined_result && img.combined_result.final_score !== undefined) {
                    aiScore = img.combined_result.final_score;
                    aiVerdict = img.combined_result.final_verdict || '--';
                } else if (img.npr_result && img.npr_result.score !== undefined) {
                    aiScore = img.npr_result.score;
                    aiVerdict = '频域噪声分析得分：' + (aiScore * 100).toFixed(1) + '%';
                }
                if (aiScore !== undefined) {
                    var aiColor = aiScore >= 0.60 ? '#dc2626' : '#16a34a';
                    html += '<div class="npr-section" style="margin-bottom:12px;">' +
                        '<div class="npr-title"><span>AI 全图生成分析</span><span class="npr-badge" style="background:' + aiColor + ';color:#fff;">' + (aiScore * 100).toFixed(1) + '%</span></div>' +
                        '<div class="npr-subtitle">' + aiVerdict + '</div></div>';
                }

                // ── 深度换脸检测（优先 deepfake_combined_result，降级到 deepfake_result）──
                var dfScore, dfVerdict;
                if (img.deepfake_combined_result && img.deepfake_combined_result.final_score !== undefined) {
                    dfScore = img.deepfake_combined_result.final_score;
                    dfVerdict = img.deepfake_combined_result.final_verdict || '--';
                } else if (img.deepfake_result && img.deepfake_result.deepfake_score !== undefined) {
                    dfScore = img.deepfake_result.deepfake_score;
                    dfVerdict = 'DeepFake 得分：' + (dfScore * 100).toFixed(1) + '%';
                }
                if (dfScore !== undefined) {
                    var dfColor = dfScore > 0.60 ? '#dc2626' : '#16a34a';
                    html += '<div class="npr-section" style="margin-bottom:12px;">' +
                        '<div class="npr-title"><span>深度换脸检测</span><span class="npr-badge" style="background:' + dfColor + ';color:#fff;">' + (dfScore * 100).toFixed(1) + '%</span></div>' +
                        '<div class="npr-subtitle">' + dfVerdict + '</div></div>';
                }

                // ── 图像篡改检测（优先 tamper_combined_result，降级到 tamper_result）──
                var tpScore, tpVerdict;
                if (img.tamper_combined_result && img.tamper_combined_result.final_score !== undefined) {
                    tpScore = img.tamper_combined_result.final_score;
                    tpVerdict = img.tamper_combined_result.final_verdict || '--';
                } else if (img.tamper_result && img.tamper_result.tamper_score !== undefined) {
                    tpScore = img.tamper_result.tamper_score;
                    tpVerdict = '篡改检测得分：' + (tpScore * 100).toFixed(1) + '%';
                }
                if (tpScore !== undefined) {
                    var tpColor = tpScore >= 0.60 ? '#dc2626' : '#16a34a';
                    html += '<div class="npr-section" style="margin-bottom:12px;">' +
                        '<div class="npr-title"><span>图像篡改检测</span><span class="npr-badge" style="background:' + tpColor + ';color:#fff;">' + (tpScore * 100).toFixed(1) + '%</span></div>' +
                        '<div class="npr-subtitle">' + tpVerdict + '</div></div>';
                }

                imageDetailBody.innerHTML = html;
                imageDetailOverlay.style.display = 'flex';
            } else {
                alert('获取图片详情失败');
            }
        })
        .catch(function (err) {
            alert('请求异常：' + err.message);
        });
    }

});
