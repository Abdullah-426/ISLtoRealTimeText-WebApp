import { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { useStore } from './store';

// Define connections manually - these are the correct MediaPipe connections
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],  // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],  // Index finger
    [0, 9], [9, 10], [10, 11], [11, 12],  // Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky
];

const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [11, 23], [12, 14], [12, 24], [13, 15], [14, 16],
    [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22], [17, 19],
    [18, 20], [19, 21], [20, 22], [23, 24], [23, 25], [24, 26], [25, 27],
    [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [31, 32]
];

// Simplified face connections for key facial features
const FACE_CONNECTIONS = [
    // Face outline (simplified)
    [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
    // Left eyebrow (simplified)
    [70, 63], [63, 105], [105, 66], [66, 107], [107, 55], [55, 65], [65, 52], [52, 53], [53, 46], [46, 70],
    // Right eyebrow (simplified)
    [296, 334], [334, 293], [293, 300], [300, 276], [276, 283], [283, 282], [282, 295], [295, 285], [285, 336], [336, 296],
    // Left eye (simplified)
    [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133], [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 161], [161, 246], [246, 33],
    // Right eye (simplified)
    [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390], [390, 249], [249, 263], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362],
    // Nose (simplified)
    [1, 2], [2, 5], [5, 4], [4, 6], [6, 19], [19, 20], [20, 94], [94, 125], [125, 141], [141, 235], [235, 236], [236, 3], [3, 51], [51, 48], [48, 115], [115, 131], [131, 134], [134, 102], [102, 49], [49, 220], [220, 305], [305, 281], [281, 360], [360, 279], [279, 331], [331, 294], [294, 358], [358, 327], [327, 326], [326, 2],
    // Mouth outer (simplified)
    [61, 84], [84, 17], [17, 314], [314, 405], [405, 320], [320, 307], [307, 375], [375, 321], [321, 308], [308, 324], [324, 318], [318, 13], [13, 82], [82, 81], [81, 80], [80, 78], [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308], [308, 61]
];

let hands: HandLandmarker | null = null;
let pose: PoseLandmarker | null = null;
let face: FaceLandmarker | null = null;

// EXACT copy of extract_2hand_keypoints function from your working app.py
function extract_2hand_keypoints(hr: any): Float32Array {
    const NUM_LM_PER_HAND = 21;
    const DIMS_PER_LM = 3;
    const FEAT_DIM = NUM_LM_PER_HAND * DIMS_PER_LM * 2; // 126

    const vec_left = new Float32Array(NUM_LM_PER_HAND * DIMS_PER_LM);
    const vec_right = new Float32Array(NUM_LM_PER_HAND * DIMS_PER_LM);

    // Initialize with zeros (exact match to app.py)
    vec_left.fill(0);
    vec_right.fill(0);

    if (!hr || !hr.landmarks || hr.landmarks.length === 0) {
        const out = new Float32Array(FEAT_DIM);
        out.set(vec_left, 0);
        out.set(vec_right, NUM_LM_PER_HAND * DIMS_PER_LM);
        return out;
    }

    // Collect handedness (exact match to app.py logic)
    const handMap: { [key: string]: any } = {};

    if (hr.handednesses && hr.handednesses.length > 0) {
        // Use handedness information (exact match to app.py)
        for (let i = 0; i < hr.landmarks.length; i++) {
            const lm = hr.landmarks[i];
            const handedness = hr.handednesses[i];
            const label = handedness?.[0]?.categoryName || (i === 0 ? 'Right' : 'Left');
            handMap[label] = lm;
        }
    } else {
        // Fallback: assign order (exact match to app.py)
        if (hr.landmarks.length >= 1) {
            handMap["Right"] = hr.landmarks[0];
        }
        if (hr.landmarks.length >= 2) {
            handMap["Left"] = hr.landmarks[1];
        }
    }

    // Fill arrays with normalized coords (exact match to app.py)
    if ("Left" in handMap) {
        const lm = handMap["Left"];
        for (let i = 0; i < Math.min(NUM_LM_PER_HAND, lm.length); i++) {
            vec_left[i * 3 + 0] = lm[i].x;  // MediaPipe already outputs [0,1] range
            vec_left[i * 3 + 1] = lm[i].y;  // MediaPipe already outputs [0,1] range
            vec_left[i * 3 + 2] = lm[i].z;  // MediaPipe z coordinate
        }
    }

    if ("Right" in handMap) {
        const lm = handMap["Right"];
        for (let i = 0; i < Math.min(NUM_LM_PER_HAND, lm.length); i++) {
            vec_right[i * 3 + 0] = lm[i].x;  // MediaPipe already outputs [0,1] range
            vec_right[i * 3 + 1] = lm[i].y;  // MediaPipe already outputs [0,1] range
            vec_right[i * 3 + 2] = lm[i].z;  // MediaPipe z coordinate
        }
    }

    // Concatenate Left(63) then Right(63) - exact match to app.py
    const out = new Float32Array(FEAT_DIM);
    out.set(vec_left, 0);
    out.set(vec_right, NUM_LM_PER_HAND * DIMS_PER_LM);
    return out;
}

// Function to reset MediaPipe models
export function resetVision() {
    hands = null;
    pose = null;
    face = null;
    console.log('MediaPipe models reset');
}


// Manual drawing functions for connecting lines
function drawHandConnections(ctx: CanvasRenderingContext2D, landmarks: any[], width: number, height: number) {
    ctx.strokeStyle = '#ffffff'; // White lines like in the image
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (const connection of HAND_CONNECTIONS) {
        const start = landmarks[connection[0]];
        const end = landmarks[connection[1]];
        if (start && end) {
            ctx.moveTo(start.x * width, start.y * height);
            ctx.lineTo(end.x * width, end.y * height);
        }
    }
    ctx.stroke();
}

function drawPoseConnections(ctx: CanvasRenderingContext2D, landmarks: any[], width: number, height: number) {
    ctx.strokeStyle = '#ffffff'; // White lines like in the image
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (const connection of POSE_CONNECTIONS) {
        const start = landmarks[connection[0]];
        const end = landmarks[connection[1]];
        if (start && end) {
            ctx.moveTo(start.x * width, start.y * height);
            ctx.lineTo(end.x * width, end.y * height);
        }
    }
    ctx.stroke();
}

function drawFaceConnections(ctx: CanvasRenderingContext2D, landmarks: any[], width: number, height: number) {
    ctx.strokeStyle = '#ffffff'; // White lines like in the image
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (const connection of FACE_CONNECTIONS) {
        const start = landmarks[connection[0]];
        const end = landmarks[connection[1]];
        if (start && end) {
            ctx.moveTo(start.x * width, start.y * height);
            ctx.lineTo(end.x * width, end.y * height);
        }
    }
    ctx.stroke();
}

export async function setupVision() {
    if (hands && pose && face) return;
    console.log('Setting up MediaPipe models...');
    const files = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    console.log('FilesetResolver loaded');

    // Use EXACT same MediaPipe settings as your working app.py
    hands = await HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' },
        numHands: 2,
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.50,  // EXACT match to app.py
        minTrackingConfidence: 0.50    // EXACT match to app.py
    });
    console.log('HandLandmarker loaded');

    pose = await PoseLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    console.log('PoseLandmarker loaded');

    face = await FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' },
        runningMode: 'VIDEO',
        numFaces: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    console.log('FaceLandmarker loaded');

    console.log('All MediaPipe models loaded successfully!');
}

export async function runVisionFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    if (!hands || !pose || !face) {
        console.log('MediaPipe models not ready yet');
        return { vec126: new Float32Array(126), vec1662: new Float32Array(1662), presenceRatio: 0 };
    }

    const w = video.videoWidth;
    const h = video.videoHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, w, h);
    const draw = new DrawingUtils(ctx as any);

    const now = performance.now();
    const hr = hands.detectForVideo(video, now);
    const pr = pose.detectForVideo(video, now);
    const fr = face.detectForVideo(video, now);

    console.log('Detection results:', {
        hands: hr.landmarks?.length || 0,
        pose: pr.landmarks?.length || 0,
        face: fr.faceLandmarks?.length || 0
    });

    // Get current mode from store
    const currentMode = useStore.getState().mode;

    // Draw hand landmarks and connections (always show for all modes)
    if (hr.landmarks && hr.landmarks.length > 0) {
        for (let i = 0; i < hr.landmarks.length; i++) {
            const lm = hr.landmarks[i];
            // Draw hand landmarks as red dots (like in the image)
            draw.drawLandmarks(lm, { color: '#ff0000', radius: 3 });
            // Draw hand connections manually
            drawHandConnections(ctx, lm, w, h);
        }
    }

    // Draw pose landmarks and connections (only for phrases and ensemble modes)
    if ((currentMode === 'phrases' || currentMode === 'ensemble') && pr.landmarks && pr.landmarks.length > 0) {
        const plm = pr.landmarks[0];
        // Draw pose landmarks as red dots (like in the image)
        draw.drawLandmarks(plm, { color: '#ff0000', radius: 3 });
        // Draw pose connections manually
        drawPoseConnections(ctx, plm, w, h);
    }

    // Draw face landmarks and connections (only for phrases and ensemble modes)
    if ((currentMode === 'phrases' || currentMode === 'ensemble') && fr.faceLandmarks && fr.faceLandmarks.length > 0) {
        const flm = fr.faceLandmarks[0];
        // Draw face landmarks as red dots (like in the image)
        draw.drawLandmarks(flm, { color: '#ff0000', radius: 1 });
        // Draw face connections manually
        drawFaceConnections(ctx, flm, w, h);
    }

    // Extract feature vectors for ML models - EXACT copy of app.py extract_2hand_keypoints function
    const vec126 = extract_2hand_keypoints(hr);

    // 1662-D holistic (pose + face + left hand + right hand)
    const poseV = new Float32Array(33 * 4);
    const faceV = new Float32Array(468 * 3);
    const lhV = new Float32Array(21 * 3);
    const rhV = new Float32Array(21 * 3);

    // Pose landmarks
    if (pr.landmarks && pr.landmarks.length > 0) {
        const plm = pr.landmarks[0];
        for (let i = 0; i < Math.min(33, plm.length); i++) {
            const lm = plm[i];
            poseV[i * 4 + 0] = lm.x;
            poseV[i * 4 + 1] = lm.y;
            poseV[i * 4 + 2] = lm.z;
            poseV[i * 4 + 3] = pr.worldLandmarks?.[0]?.[i]?.visibility || 1.0;
        }
    }

    // Face landmarks
    if (fr.faceLandmarks && fr.faceLandmarks.length > 0) {
        const flm = fr.faceLandmarks[0];
        for (let i = 0; i < Math.min(468, flm.length); i++) {
            const lm = flm[i];
            faceV[i * 3 + 0] = lm.x;
            faceV[i * 3 + 1] = lm.y;
            faceV[i * 3 + 2] = lm.z;
        }
    }

    // Left hand landmarks (use same data as vec126 - exact match to app.py)
    for (let j = 0; j < 21; j++) {
        lhV[j * 3 + 0] = vec126[j * 3 + 0];  // Same normalized [0,1] coordinates
        lhV[j * 3 + 1] = vec126[j * 3 + 1];  // Same normalized [0,1] coordinates
        lhV[j * 3 + 2] = vec126[j * 3 + 2];  // Same z coordinates
    }

    // Right hand landmarks (use same data as vec126 - exact match to app.py)
    for (let j = 0; j < 21; j++) {
        rhV[j * 3 + 0] = vec126[63 + j * 3 + 0];  // Same normalized [0,1] coordinates
        rhV[j * 3 + 1] = vec126[63 + j * 3 + 1];  // Same normalized [0,1] coordinates
        rhV[j * 3 + 2] = vec126[63 + j * 3 + 2];  // Same z coordinates
    }

    const vec1662 = new Float32Array(1662);
    vec1662.set(poseV, 0);
    vec1662.set(faceV, 33 * 4);
    vec1662.set(lhV, 33 * 4 + 468 * 3);
    vec1662.set(rhV, 33 * 4 + 468 * 3 + 21 * 3);

    // Presence ratio
    let nz = 0;
    for (let i = 0; i < vec1662.length; i++) {
        if (vec1662[i] !== 0) nz++;
    }
    const presenceRatio = nz / vec1662.length;

    return { vec126, vec1662, presenceRatio };
}
