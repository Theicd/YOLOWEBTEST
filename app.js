// מערכת זיהוי אובייקטים עם YOLO11

// התייחסות לאלמנטים בדף
let dropZone;
let uploadInput;
let preview; 
let previewContainer;
let modelStatusElement;
let loading;
let loadingText;
let resultsList;

// משתנים גלובליים
let session = null;
let isModelLoaded = false;
let lastResults = []; // שמירת התוצאות האחרונות

// נתיבים אפשריים למודל - ננסה לפי הסדר
const modelPaths = [
    './yolo11n.onnx', // נתיב יחסי מקומי
    '/yolo11n.onnx', // נתיב שורש
    'https://github.com/Theicd/YOLOWEBTEST/raw/main/yolo11n.onnx', // נתיב גיטהאב ישיר
    'https://raw.githubusercontent.com/Theicd/YOLOWEBTEST/main/yolo11n.onnx', // נתיב raw.githubusercontent.com
    './yolov8n-seg.onnx', // גיבוי - נתיב יחסי מקומי לגרסה קודמת
    'https://github.com/Theicd/YOLOWEBTEST/raw/main/yolov8n-seg.onnx' // גיבוי - נתיב גיטהאב לגרסה קודמת
];

// קבועים למודל
const IMAGE_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.5; // העלאת סף הביטחון - היה 0.25 קודם
const NUM_CLASSES = 80;
const ANCHORS = [];

// רשימת המחלקות של COCO (למודל YOLO)
const COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

// שמות מחלקות בעברית (אופציונלי)
const COCO_CLASSES_HEBREW = [
    'אדם', 'אופניים', 'מכונית', 'אופנוע', 'מטוס', 'אוטובוס', 'רכבת', 'משאית', 'סירה',
    'רמזור', 'ברז כיבוי', 'תמרור עצור', 'מד חניה', 'ספסל', 'ציפור', 'חתול', 'כלב',
    'סוס', 'כבשה', 'פרה', 'פיל', 'דוב', 'זברה', 'ג׳ירפה', 'תיק גב', 'מטריה',
    'תיק יד', 'עניבה', 'מזוודה', 'פריסבי', 'מגלשיים', 'סנובורד', 'כדור ספורט', 'עפיפון',
    'מחבט בייסבול', 'כפפת בייסבול', 'סקייטבורד', 'גלשן', 'מחבט טניס', 'בקבוק',
    'כוס יין', 'כוס', 'מזלג', 'סכין', 'כף', 'קערה', 'בננה', 'תפוח', 'כריך',
    'תפוז', 'ברוקולי', 'גזר', 'נקניקיה', 'פיצה', 'דונאט', 'עוגה', 'כיסא', 'ספה',
    'עציץ', 'מיטה', 'שולחן אוכל', 'שירותים', 'טלוויזיה', 'לפטופ', 'עכבר', 'שלט',
    'מקלדת', 'טלפון נייד', 'מיקרוגל', 'תנור', 'טוסטר', 'כיור', 'מקרר', 'ספר',
    'שעון', 'אגרטל', 'מספריים', 'דובי', 'מייבש שיער', 'מברשת שיניים'
];

// רשימת מחלקות מועדפות לתצוגה (מסודרות לפי סדר עדיפות)
const PREFERRED_CLASSES = [
    'banana', 'apple', 'orange', 'broccoli', 'carrot', // פירות וירקות
    'person', 'dog', 'cat', // בעלי חיים ואנשים
    'chair', 'couch', 'dining table', // רהיטים
    'car', 'truck', 'bicycle' // כלי רכב
];

// אתחול האפליקציה
document.addEventListener('DOMContentLoaded', function() {
    // התייחסות לאלמנטים בדף
    dropZone = document.getElementById('drop-zone');
    uploadInput = document.getElementById('upload-input');
    preview = document.getElementById('preview');
    previewContainer = document.getElementById('preview-container');
    modelStatusElement = document.getElementById('modelStatus');
    loading = document.getElementById('loading');
    loadingText = document.getElementById('loading-text');
    resultsList = document.getElementById('results-list');
    
    // אתחול פעולות
    init();
});

// אתחול פונקציות ואירועים
function init() {
    console.log("אתחול האפליקציה התחיל");
    
    // הגדרת מאזיני אירועים
    setupEventListeners();
    
    // טעינה אוטומטית של המודל המובנה
    loadBuiltInModel();
}

// הגדרת מאזיני אירועים
function setupEventListeners() {
    // אירוע גרירת קובץ על אזור הגרירה
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });

    // אירוע לחיצה על אזור הגרירה
    dropZone.addEventListener('click', () => {
        uploadInput.click();
    });

    // אירוע בחירת קובץ
    uploadInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageUpload(e.target.files[0]);
        }
    });

    // אירוע לניקוי תוצאות
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearResults);
    }
    
    // אירוע לייצוא תוצאות
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportResults);
    }
    
    // הדפסת כל האלמנטים לבדיקה
    console.log("אלמנטים שנטענו:", {
        dropZone,
        uploadInput,
        preview,
        previewContainer,
        modelStatusElement,
        loading,
        loadingText,
        resultsList
    });
}

// טעינת המודל המובנה
function loadBuiltInModel() {
    // עדכון סטטוס הטעינה
    updateModelStatus('loading', 'טוען מודל... (זה עשוי לקחת מספר שניות)');
    
    // טעינת המודל - ננסה לפי הסדר
    tryLoadModelFromPaths(0);
}

// ניסיון טעינת מודל מכל הנתיבים האפשריים
async function tryLoadModelFromPaths(index) {
    if (index >= modelPaths.length) {
        // נכשלנו לטעון מכל הנתיבים
        updateModelStatus('error', 'נכשל בטעינת המודל מכל המקורות האפשריים');
        console.error('נכשל לטעון את המודל מכל הנתיבים האפשריים');
        return;
    }
    
    const currentPath = modelPaths[index];
    console.log(`מנסה לטעון מודל מנתיב (${index + 1}/${modelPaths.length}): ${currentPath}`);
    
    try {
        await loadModel(currentPath);
        // אם הגענו לכאן, המודל נטען בהצלחה
    } catch (error) {
        console.warn(`שגיאה בטעינת המודל מנתיב ${currentPath}:`, error);
        console.log(`מנסה נתיב הבא (${index + 1} מתוך ${modelPaths.length})`);
        // ניסיון הנתיב הבא
        tryLoadModelFromPaths(index + 1);
    }
}

// טעינת המודל
async function loadModel(modelPath) {
    try {
        console.log('התחלת טעינת המודל:', modelPath);
        
        // הגדרת אפשרויות לטעינת המודל
        const options = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };
        
        // ניסיון לטעון את המודל
        if (modelPath.startsWith('http')) {
            // טעינה מ-URL מרוחק
            console.log('מנסה לטעון מודל מ-URL מרוחק:', modelPath);
            
            try {
                // ניסיון ראשון - טעינה ישירה (עשוי להיכשל בגלל CORS)
                session = await ort.InferenceSession.create(modelPath, options);
            } catch (corsError) {
                console.warn('שגיאת CORS בטעינה ישירה, מנסה דרך fetch:', corsError.message);
                
                // ניסיון שני - שימוש ב-fetch
                updateModelStatus('loading', 'מוריד את המודל... (יכול לקחת מספר שניות)');
                
                const response = await fetch(modelPath);
                
                if (!response.ok) {
                    throw new Error(`שגיאה בהורדת המודל: ${response.status} ${response.statusText}`);
                }
                
                updateModelStatus('loading', 'מעבד את המודל...');
                const modelData = await response.arrayBuffer();
                
                // טעינת המודל מה-ArrayBuffer
                session = await ort.InferenceSession.create(new Uint8Array(modelData), options);
            }
        } else {
            // טעינה מקומית
            console.log('מנסה לטעון מודל מקומי:', modelPath);
            session = await ort.InferenceSession.create(modelPath, options);
        }
        
        console.log("המודל נטען בהצלחה:", session);
        
        // בדיקת מבנה קלט/פלט של המודל
        const inputNames = session.inputNames;
        const outputNames = session.outputNames;
        console.log("קלטי המודל:", inputNames);
        console.log("פלטי המודל:", outputNames);
        
        // עדכון הסטטוס
        isModelLoaded = true;
        updateModelStatus('loaded', 'המודל נטען בהצלחה!');
        console.log('המודל נטען בהצלחה');
        
        return true; // חשוב להחזיר הצלחה עבור פונקציית tryLoadModelFromPaths
        
    } catch (error) {
        console.error('שגיאה בטעינת המודל:', error);
        // לא מעדכנים את הסטטוס כאן כדי לאפשר ניסיון נתיבים נוספים
        console.error('פרטי השגיאה:', error.message);
        
        // איפוס משתנים
        isModelLoaded = false;
        session = null;
        
        // מעבירים את השגיאה כדי שפונקציית tryLoadModelFromPaths תנסה את הנתיב הבא
        throw error;
    }
}

// עדכון סטטוס טעינת המודל
function updateModelStatus(status, message) {
    if (!modelStatusElement) {
        console.error('אלמנט סטטוס המודל לא נמצא!');
        return;
    }
    
    // אם האלמנטים הפנימיים לא קיימים עדיין, נייצר אותם
    let statusLoader = modelStatusElement.querySelector('.loader');
    let statusText = modelStatusElement.querySelector('p');
    
    if (!statusLoader) {
        statusLoader = document.createElement('div');
        statusLoader.className = 'loader';
        modelStatusElement.appendChild(statusLoader);
    }
    
    if (!statusText) {
        statusText = document.createElement('p');
        modelStatusElement.appendChild(statusText);
    }
    
    // מציג את האינדיקטור
    modelStatusElement.style.display = 'flex';
    
    switch (status) {
        case 'loading':
            statusLoader.style.display = 'block';
            statusText.textContent = message || 'טוען את מודל YOLO...';
            modelStatusElement.className = 'model-status loading';
            break;
        case 'loaded':
            statusLoader.style.display = 'none';
            statusText.textContent = 'המודל נטען בהצלחה!';
            
            // הסתרת ההודעה אחרי שניה
            setTimeout(() => {
                modelStatusElement.style.display = 'none';
            }, 2000);
            
            modelStatusElement.className = 'model-status success';
            break;
        case 'error':
            statusLoader.style.display = 'none';
            statusText.textContent = message || 'שגיאה בטעינת המודל';
            modelStatusElement.className = 'model-status error';
            break;
        case 'waiting':
            statusLoader.style.display = 'none';
            statusText.textContent = message || 'לחץ על "טען מודל" כדי להתחיל';
            modelStatusElement.className = 'model-status waiting';
            break;
    }
}

// הצגת הודעת שגיאה
function showError(message) {
    // בדיקה שזאת תמונה
    if (!file.type.startsWith('image/')) {
        throw new Error('אנא העלה קובץ תמונה בלבד');
    }
    
    // בדיקה שהמודל נטען
    if (!isModelLoaded) {
        throw new Error('אנא טען מודל לפני העלאת תמונה');
    }
        
    // בדיקה אם יש כבר אלמנט שגיאה
    let errorElement = document.getElementById('error-message');
    
    // אם אין אלמנט שגיאה, יצירת אחד חדש
    if (!errorElement) {
        errorElement = document.createElement('div');
        errorElement.id = 'error-message';
        errorElement.className = 'error-message';
        document.body.appendChild(errorElement);
    }
    
    // הגדרת הודעת השגיאה
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    
    // הסתרת ההודעה אחרי 5 שניות
    setTimeout(() => {
        errorElement.style.display = 'none';
    }, 5000);
    
    console.error(message);
}

// הצגת אינדיקטור טעינה
function showLoading(message = 'טוען...') {
    // הצגת אינדיקטור הטעינה
    if (loading) {
        loading.style.display = 'flex';
    }
    
    // הצגת הודעת טעינה
    if (loadingText) {
        loadingText.textContent = message;
    }
    
    console.log(`הצגת אינדיקטור טעינה: ${message}`);
}

// הסתרת אינדיקטור טעינה
function hideLoading() {
    // הסתרת אינדיקטור הטעינה
    if (loading) {
        loading.style.display = 'none';
    }
    
    console.log('הסתרת אינדיקטור טעינה');
}

// טעינת תמונה כאלמנט
function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                resolve(img);
            };
            img.onerror = function() {
                reject(new Error('שגיאה בטעינת התמונה'));
            };
            img.src = e.target.result;
        };
        
        reader.onerror = function() {
            reject(new Error('שגיאה בקריאת הקובץ'));
        };
        
        reader.readAsDataURL(file);
    });
}

// טיפול בהעלאת תמונה
async function handleImageUpload(file) {
    console.log("התחיל טיפול בתמונה:", file.name);
    
    try {
        // בדיקה שזאת תמונה
        if (!file.type.startsWith('image/')) {
            throw new Error('אנא העלה קובץ תמונה בלבד');
        }
        
        // בדיקה שהמודל נטען
        if (!isModelLoaded) {
            throw new Error('אנא טען מודל לפני העלאת תמונה');
        }
        
        // טעינת התמונה כאלמנט
        const img = await loadImage(file);
        console.log("התמונה נטענה כאלמנט:", img.width, "x", img.height);
        
        // ניקוי תוצאות
        clearResults();
        
        // הצגת התמונה בתצוגה המקדימה
        displayImage(img);
        
        // עיבוד התמונה
        await processImage(img);
        
    } catch (error) {
        console.error("שגיאה בטיפול בתמונה:", error);
        showError(error.message);
        hideLoading();
    }
}

// עיבוד תמונה
async function processImage(img) {
    try {
        // בדיקה שהמודל טעון
        if (!session || !isModelLoaded) {
            throw new Error('המודל לא נטען, אנא טען מודל לפני עיבוד תמונה');
        }
        
        console.log("מתחיל עיבוד תמונה...");
        
        // הצגת אינדיקטור טעינה
        showLoading("מעבד תמונה...");
        
        // הכנת התמונה לעיבוד
        const preprocessedData = await preprocessImageForYOLO(img);
        
        if (!preprocessedData || !preprocessedData.tensor) {
            throw new Error('שגיאה בהכנת התמונה לעיבוד');
        }
        
        console.log("התמונה הוכנה לעיבוד:", preprocessedData);
        
        // הרצת המודל
        const feeds = {};
        feeds[session.inputNames[0]] = preprocessedData.tensor;
        
        console.log(`מריץ מודל עם קלט בשם: ${session.inputNames[0]}`);
        
        // הפעלת המודל
        const outputMap = await session.run(feeds);
        console.log("המודל הסתיים, פלט:", outputMap);
        
        // פענוח תוצאות המודל
        const boxes = decodeYoloV8Output(outputMap, preprocessedData.dimensions);
        
        // הצגת התוצאות
        if (boxes && boxes.length > 0) {
            // שליחת התוצאות לתצוגה
            displayResults(boxes, preprocessedData.originalSize);
            
            // שמירת התוצאות לייצוא
            lastResults = boxes;
            
            console.log(`זוהו ${boxes.length} אובייקטים`);
        } else {
            console.log('לא זוהו אובייקטים בתמונה');
            resultsList.innerHTML = '<li class="no-results">לא זוהו אובייקטים בתמונה</li>';
        }
        
        // הסתרת אינדיקטור טעינה
        hideLoading();
    } catch (error) {
        console.error("שגיאה בעיבוד התמונה:", error);
        showError(`שגיאה בעיבוד התמונה: ${error.message}`);
        hideLoading();
    }
}

// פענוח פלט של YOLO11
function decodeYoloV8Output(outputMap, imgDimensions) {
    const boxes = [];
    const outputKeys = Object.keys(outputMap);
    const boxData = outputMap[outputKeys[0]].data;
    const [batchSize, numDetections, outputDims] = outputMap[outputKeys[0]].dims;

    // בדיקה אם יש הבדל במבנה הפלט בין YOLOv8 ל-YOLO11
    const isYOLO11 = outputDims === 84; // YOLO11 יש לו 84 ערכים לכל זיהוי (4 קואורדינטות + 80 מחלקות)
    const confidenceIndex = isYOLO11 ? 4 : 4; // אותו אינדקס לביטחון בשני המודלים
    const classStartIndex = isYOLO11 ? 4 : 5; // תחילת אינדקס המחלקות
    const numClasses = isYOLO11 ? 80 : 80; // מספר המחלקות

    console.log(`מפענח פלט YOLO מסוג: ${isYOLO11 ? 'YOLO11' : 'YOLOv8'}, מימדים: ${outputDims}`);

    for (let i = 0; i < numDetections; i++) {
        const offset = i * outputDims;
        // בYOLO11 אנחנו בודקים את הביטחון הכללי בצורה שונה
        let confidence = 0;
        
        if (isYOLO11) {
            // בYOLO11 הביטחון מחושב ישירות מהציונים של המחלקות
            let maxClassScore = 0;
            for (let c = 0; c < numClasses; c++) {
                const score = boxData[offset + classStartIndex + c];
                maxClassScore = Math.max(maxClassScore, score);
            }
            confidence = maxClassScore; // הביטחון הגבוה ביותר מבין המחלקות
        } else {
            // בYOLOv8 הביטחון נמצא באינדקס ספציפי
            confidence = boxData[offset + confidenceIndex];
        }

        if (confidence > CONFIDENCE_THRESHOLD) {
            let maxClassScore = 0, maxClassIndex = 0;
            
            // מציאת המחלקה עם הציון הגבוה ביותר
            for (let c = 0; c < numClasses; c++) {
                const score = boxData[offset + classStartIndex + c];
                if (score > maxClassScore) {
                    maxClassScore = score;
                    maxClassIndex = c;
                }
            }
            
            const finalScore = isYOLO11 ? maxClassScore : confidence * maxClassScore;
            
            if (finalScore > CONFIDENCE_THRESHOLD) {
                const cx = boxData[offset];
                const cy = boxData[offset + 1];
                const width = boxData[offset + 2];
                const height = boxData[offset + 3];
                
                // סינון אובייקטים קטנים מדי (פחות מ-1% משטח התמונה)
                const boxArea = width * height;
                const MIN_AREA_RATIO = 0.01;
                
                if (boxArea > MIN_AREA_RATIO) {
                    boxes.push({
                        xmin: (cx - width / 2) * imgDimensions[0],
                        ymin: (cy - height / 2) * imgDimensions[1],
                        xmax: (cx + width / 2) * imgDimensions[0],
                        ymax: (cy + height / 2) * imgDimensions[1],
                        class: maxClassIndex,
                        className: COCO_CLASSES[maxClassIndex],
                        classNameHebrew: COCO_CLASSES_HEBREW[maxClassIndex],
                        score: finalScore,
                        isPreferred: PREFERRED_CLASSES.includes(COCO_CLASSES[maxClassIndex])
                    });
                }
            }
        }
    }
    
    // מיון התוצאות:
    // 1. מחלקות מועדפות לפני מחלקות אחרות
    // 2. לפי רמת הביטחון (מהגבוה לנמוך)
    boxes.sort((a, b) => {
        if (a.isPreferred && !b.isPreferred) return -1;
        if (!a.isPreferred && b.isPreferred) return 1;
        return b.score - a.score;
    });
    
    // מגביל את מספר האובייקטים המוצגים
    const MAX_OBJECTS = 10;
    return boxes.slice(0, MAX_OBJECTS);
}

// עיבוד התמונה למודל YOLO
async function preprocessImageForYOLO(img) {
    try {
        console.log("מתחיל עיבוד מקדים של התמונה למודל YOLO");
        
        // בהרבה מודלים של YOLO, הקלט הוא 640x640
        const inputWidth = IMAGE_SIZE;
        const inputHeight = IMAGE_SIZE;
        
        console.log(`מימדי קלט למודל: ${inputWidth}x${inputHeight}`);
        
        // יצירת קנבס לשינוי גודל התמונה
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = inputWidth;
        canvas.height = inputHeight;
        
        // ציור התמונה על הקנבס בגודל החדש
        ctx.drawImage(img, 0, 0, inputWidth, inputHeight);
        
        // קבלת נתוני הפיקסלים
        const imageData = ctx.getImageData(0, 0, inputWidth, inputHeight);
        const data = imageData.data;
        
        // אתחול מערכים עבור הערוצים RGB
        const redChannel = new Float32Array(inputWidth * inputHeight);
        const greenChannel = new Float32Array(inputWidth * inputHeight);
        const blueChannel = new Float32Array(inputWidth * inputHeight);
        
        // המרת נתוני התמונה לפורמט המתאים למודל YOLO
        // YOLO11 וגם YOLOv8 מצפים לתמונה בפורמט [batch, channels, height, width] עם נירמול לטווח [0,1]
        for (let i = 0; i < data.length / 4; i++) {
            // נרמול הערכים לטווח [0,1]
            redChannel[i] = data[i * 4] / 255.0;
            greenChannel[i] = data[i * 4 + 1] / 255.0;
            blueChannel[i] = data[i * 4 + 2] / 255.0;
        }
        
        // יצירת מערך מאוחד לפי סדר CHW (channels, height, width)
        const rgbData = new Float32Array(3 * inputHeight * inputWidth);
        
        // העתקת הנתונים למערך המאוחד
        rgbData.set(redChannel, 0);
        rgbData.set(greenChannel, inputWidth * inputHeight);
        rgbData.set(blueChannel, 2 * inputWidth * inputHeight);
        
        // יצירת טנסור עבור ה-ONNX runtime
        const tensor = new ort.Tensor('float32', rgbData, [1, 3, inputHeight, inputWidth]);
        
        console.log("עיבוד מקדים של התמונה הושלם");
        
        // החזרת הטנסור ומימדי הקלט
        return {
            tensor: tensor,
            dimensions: [1, 3, inputHeight, inputWidth],
            originalSize: [img.width, img.height]
        };
    } catch (error) {
        console.error("שגיאה בעיבוד מקדים של התמונה:", error);
        throw new Error(`שגיאה בעיבוד מקדים של התמונה: ${error.message}`);
    }
}

// הצגת תמונה בתצוגה המקדימה
function displayImage(img) {
    // ניקוי תצוגה קודמת
    preview.innerHTML = '';
    
    // הוספת התמונה לתצוגה
    preview.appendChild(img);
    
    // הצגת מיכל התמונה
    previewContainer.style.display = 'grid';
    
    console.log("התמונה הוצגה בתצוגה המקדימה");
}

// ניקוי תוצאות קודמות
function clearResults() {
    // ניקוי תצוגת תמונה
    preview.innerHTML = '';
    
    // ניקוי רשימת תוצאות
    if (resultsList) {
        resultsList.innerHTML = '';
    }
    
    // ניקוי משתנה תוצאות אחרונות
    lastResults = [];
    
    console.log("תוצאות קודמות נוקו");
}

// הצגת תוצאות הזיהוי
function displayResults(boxes, dimensions) {
    // התוצאות כבר ממוינות בשלב הפענוח
    
    // שמירת התוצאות
    lastResults = boxes;
    
    // הצגת התוצאות
    const [imageWidth, imageHeight] = dimensions;
    
    // ציור התיבות על גבי התמונה
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    
    // קבלת קונטקסט ציור
    ctx.drawImage(preview.firstChild, 0, 0, imageWidth, imageHeight);
    
    // ציור התיבות
    displayBoxes(ctx, boxes, dimensions);
    
    // הצגת התמונה עם התיבות
    const resultImage = document.createElement('img');
    resultImage.src = canvas.toDataURL();
    resultImage.width = imageWidth;
    resultImage.height = imageHeight;
    
    // החלפת התמונה בתצוגה
    preview.innerHTML = '';
    preview.appendChild(resultImage);
    
    // ניקוי רשימת התוצאות
    resultsList.innerHTML = '';
    
    // הוספת כל תוצאה לרשימה
    for (let i = 0; i < boxes.length; i++) {
        addResultItem(boxes[i], i);
    }
    
    console.log(`הוצגו ${boxes.length} תוצאות זיהוי`);
}

// הצגת תיבות על גבי התמונה
function displayBoxes(ctx, boxes, dimensions) {
    // הגדרת צבעים
    const colors = ['#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'];
    
    // ציור תיבות ותוויות
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        
        // חישוב מיקום בפיקסלים
        const x = box.xmin;
        const y = box.ymin;
        const width = box.xmax - box.xmin;
        const height = box.ymax - box.ymin;
        
        // בחירת צבע לפי המחלקה
        const color = colors[box.class % colors.length];
        
        // הגדרת סגנון לתיבה
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        
        // ציור התיבה
        ctx.strokeRect(x, y, width, height);
        
        // הגדרת סגנון לטקסט
        ctx.fillStyle = color;
        ctx.font = '18px Arial';
        
        // קבלת שם המחלקה בעברית
        const className = box.classNameHebrew || 'לא ידוע';
        
        // הכנת הטקסט לתווית - שם המחלקה ואחוז ביטחון
        const label = `${className}: ${Math.round(box.score * 100)}%`;
        
        // מדידת אורך הטקסט
        const textWidth = ctx.measureText(label).width;
        
        // ציור רקע לטקסט
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);
        
        // ציור הטקסט
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x + 5, y - 7);
    }
}

// הוספת פריט לרשימת התוצאות
function addResultItem(box, index) {
    // יצירת פריט ברשימה
    const listItem = document.createElement('li');
    
    // קביעת צבע רקע לפי המחלקה
    const colors = ['#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'];
    const color = colors[box.class % colors.length];
    
    // קבלת שם המחלקה בעברית
    const className = box.classNameHebrew || 'לא ידוע';
    
    // יצירת HTML לפריט
    listItem.innerHTML = `
        <div class="result-item" style="border-color: ${color}">
            <div class="result-header" style="background-color: ${color}">
                <span class="result-class">${className}</span>
                <span class="result-score">${Math.round(box.score * 100)}%</span>
            </div>
            <div class="result-details">
                <div>מיקום: X=${Math.round(box.xmin * 100)}%-${Math.round(box.xmax * 100)}%, Y=${Math.round(box.ymin * 100)}%-${Math.round(box.ymax * 100)}%</div>
            </div>
        </div>
    `;
    
    // הוספת הפריט לרשימה
    resultsList.appendChild(listItem);
}

// יצירת צבע רנדומלי
function getRandomColor() {
    // צבעים בסיסיים עם נראות טובה
    const colors = [
        '#FF5733', // כתום-אדום
        '#33FF57', // ירוק בהיר
        '#3357FF', // כחול
        '#FF33E9', // ורוד
        '#33FFF5', // טורקיז
        '#F7FF33', // צהוב
        '#A233FF', // סגול
        '#FF3380', // ורוד-אדום
        '#33FFB2', // מנטה
        '#F96924', // כתום כהה
        '#247BF9', // כחול בהיר
        '#90F924', // ירוק-צהוב
        '#F92484', // פוקסיה
        '#24F9E0'  // תכלת
    ];
    
    // בחירת צבע אקראי מהרשימה
    return colors[Math.floor(Math.random() * colors.length)];
}

// ייצוא תוצאות כקובץ JSON
function exportResults() {
    if (lastResults.length === 0) {
        showError('אין תוצאות לייצוא');
        return;
    }
    
    // הכנת נתונים לייצוא
    const exportData = lastResults.map(result => ({
        class: result.className,
        confidence: Math.round(result.score * 100) / 100,
        bbox: {
            x_min: Math.round(result.xmin),
            y_min: Math.round(result.ymin),
            x_max: Math.round(result.xmax),
            y_max: Math.round(result.ymax)
        }
    }));
    
    // המרה ל-JSON
    const dataStr = JSON.stringify(exportData, null, 2);
    
    // יצירת קובץ
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    // יצירת קישור להורדה
    const a = document.createElement('a');
    a.href = url;
    a.download = 'yolo_detection_results.json';
    
    // הוספה לדף והפעלה
    document.body.appendChild(a);
    a.click();
    
    // ניקוי
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
