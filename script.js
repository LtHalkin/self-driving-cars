/* =====================================================================
   AUTONOMOUS DRIVING CITY SIMULATION
   Plain HTML / CSS / JavaScript. No frameworks, no libraries.
   =====================================================================
   Table of contents:
     1. Utility math / geometry
     2. Road network (nodes, edges, lanes, boundaries, checkpoints)
     3. Traffic lights
     4. Sensor
     5. Neural network (Layer, NeuralNetwork)
     6. Car (AI-controlled and scripted "traffic" cars)
     7. Evolution manager
     8. Camera
     9. World renderer (roads, buildings, trees, minimap, day/night)
    10. UI wiring + main loop
   ===================================================================== */

'use strict';

/* ============================== 1. UTILS ============================== */

function lerp(a, b, t) { return a + (b - a) * t; }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function angleOf(a, b) { return Math.atan2(b.y - a.y, b.x - a.x); }

// Returns the intersection point of two segments AB and CD, or null.
// offset (0..1) is how far along AB the hit occurred - used by sensors.
function getIntersection(A, B, C, D) {
    const tTop = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x);
    const uTop = (C.y - A.y) * (A.x - B.x) - (C.x - A.x) * (A.y - B.y);
    const bottom = (D.y - C.y) * (B.x - A.x) - (D.x - C.x) * (B.y - A.y);
    if (bottom === 0) return null;
    const t = tTop / bottom;
    const u = uTop / bottom;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        return { x: lerp(A.x, B.x, t), y: lerp(A.y, B.y, t), offset: t };
    }
    return null;
}

// True if two polygons (arrays of {x,y}) intersect (edge crossing test).
function polysIntersect(poly1, poly2) {
    for (let i = 0; i < poly1.length; i++) {
        for (let j = 0; j < poly2.length; j++) {
            const touch = getIntersection(
                poly1[i], poly1[(i + 1) % poly1.length],
                poly2[j], poly2[(j + 1) % poly2.length]
            );
            if (touch) return true;
        }
    }
    return false;
}

function pointInPoly(p, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
        const intersect = ((yi > p.y) !== (yj > p.y)) &&
            (p.x < (xj - xi) * (p.y - yi) / (yj - yi + 0.00000001) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function pointToSegmentDist(p, a, b) {
    const l2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
    if (l2 === 0) return dist(p, a);
    let t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
    t = Math.max(0, Math.min(1, t));
    const proj = { x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) };
    return dist(p, proj);
}

function randRange(min, max) { return min + Math.random() * (max - min); }
function randInt(min, max) { return Math.floor(randRange(min, max + 1)); }
function pickRandom(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function getRGBA(value) {
    // value expected in -1..1 (weight/activation visualisation helper)
    const alpha = Math.abs(value);
    const R = value < 0 ? 0 : 255;
    const G = value < 0 ? 0 : 255;
    const B = value > 0 ? 0 : 255;
    return `rgba(${R},${G},${B},${alpha})`;
}

/* ========================= 2. ROAD NETWORK ============================= */

class RoadNetwork {
    constructor(seed) {
        this.nodes = [];       // {id, x, y}
        this.edges = [];       // Road objects
        this.checkpoints = []; // {x,y,radius,order}
        this.buildings = [];   // {poly} for rendering + collision
        this.trees = [];       // {x,y,radius}
        this.parking = [];     // {poly} decorative parking lots
        this.streetLights = []; // {x,y}
        this._build();
        this._computeGeometry();
        this._placeCheckpoints();
        this._placeScenery();
    }

    _build() {
        const COLS = 7, ROWS = 6, SPACING = 480;
        const jitter = 45;
        // Create a jittered grid of nodes.
        const grid = [];
        for (let r = 0; r < ROWS; r++) {
            const row = [];
            for (let c = 0; c < COLS; c++) {
                const node = {
                    id: r * COLS + c,
                    x: c * SPACING + randRange(-jitter, jitter),
                    y: r * SPACING + randRange(-jitter, jitter),
                    r, c
                };
                this.nodes.push(node);
                row.push(node);
            }
            grid.push(row);
        }

        // Candidate edges: orthogonal neighbours + a diagonal "highway" set.
        const candidates = [];
        const key = (a, b) => a.id < b.id ? `${a.id}_${b.id}` : `${b.id}_${a.id}`;
        const added = new Set();
        const addCandidate = (a, b, kind) => {
            const k = key(a, b);
            if (added.has(k)) return;
            added.add(k);
            candidates.push({ a, b, kind });
        };
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (c < COLS - 1) addCandidate(grid[r][c], grid[r][c + 1], 'grid');
                if (r < ROWS - 1) addCandidate(grid[r][c], grid[r + 1][c], 'grid');
            }
        }
        // A diagonal highway cutting across the whole city.
        for (let i = 0; i < Math.min(ROWS, COLS) - 1; i++) {
            addCandidate(grid[i][i], grid[i + 1][i + 1], 'highway');
        }
        for (let i = 0; i < ROWS - 2; i++) {
            if (COLS - 1 - i - 1 >= 0) {
                addCandidate(grid[i][COLS - 1 - i], grid[i + 1][COLS - 1 - i - 1], 'highway');
            }
        }

        // Randomised Kruskal-ish spanning structure to guarantee connectivity.
        const parent = {};
        this.nodes.forEach(n => parent[n.id] = n.id);
        const find = id => parent[id] === id ? id : (parent[id] = find(parent[id]));
        const union = (a, b) => { parent[find(a)] = find(b); };

        const shuffled = candidates.slice().sort(() => Math.random() - 0.5);
        const chosen = [];
        for (const cand of shuffled) {
            if (find(cand.a.id) !== find(cand.b.id)) {
                union(cand.a.id, cand.b.id);
                chosen.push(cand);
            }
        }
        // Add extra edges back for cycles / realism (network richness),
        // skipping some to create dead ends and varied intersections.
        for (const cand of shuffled) {
            if (chosen.includes(cand)) continue;
            if (Math.random() < 0.45) chosen.push(cand);
        }

        // Build Road objects with widths depending on kind / degree.
        const degree = {};
        chosen.forEach(c => {
            degree[c.a.id] = (degree[c.a.id] || 0) + 1;
            degree[c.b.id] = (degree[c.b.id] || 0) + 1;
        });
        for (const cand of chosen) {
            const isHighway = cand.kind === 'highway';
            const width = isHighway ? 130 : (Math.random() < 0.3 ? 70 : 95);
            const lanesEachWay = width > 110 ? 2 : 1;
            this.edges.push(new Road(cand.a, cand.b, width, lanesEachWay, isHighway));
        }
        this.degree = degree;
    }

    _computeGeometry() {
        const NODE_RADIUS = n => {
            let maxW = 0;
            for (const e of this.edges) {
                if (e.a === n || e.b === n) maxW = Math.max(maxW, e.width);
            }
            return maxW / 2 + 6;
        };
        this.nodeRadius = {};
        this.nodes.forEach(n => this.nodeRadius[n.id] = NODE_RADIUS(n));

        for (const road of this.edges) {
            road.build(this.nodeRadius[road.a.id], this.nodeRadius[road.b.id]);
        }

        // Global list of boundary segments used by sensors / off-road checks.
        this.borders = [];
        for (const road of this.edges) {
            this.borders.push(road.leftBorder, road.rightBorder);
        }

        // Traffic lights at intersections with 3+ connected roads.
        this.trafficLights = [];
        for (const node of this.nodes) {
            const connected = this.edges.filter(e => e.a === node || e.b === node);
            if (connected.length >= 3) {
                const light = new TrafficLight(node, connected);
                this.trafficLights.push(light);
            }
        }
    }

    _placeCheckpoints() {
        // BFS order from an arbitrary spawn node gives checkpoints an
        // "order" so fitness rewards genuine forward progress through
        // the network rather than looping on one road.
        const startNode = this.nodes[0];
        const distMap = new Map();
        distMap.set(startNode.id, 0);
        const queue = [startNode];
        const adjacency = new Map();
        this.nodes.forEach(n => adjacency.set(n.id, []));
        this.edges.forEach(e => {
            adjacency.get(e.a.id).push(e.b);
            adjacency.get(e.b.id).push(e.a);
        });
        while (queue.length) {
            const cur = queue.shift();
            for (const nb of adjacency.get(cur.id)) {
                if (!distMap.has(nb.id)) {
                    distMap.set(nb.id, distMap.get(cur.id) + 1);
                    queue.push(nb);
                }
            }
        }

        let order = 0;
        for (const road of this.edges) {
            const baseOrder = (distMap.get(road.a.id) ?? 0) + (distMap.get(road.b.id) ?? 0);
            for (const wp of road.waypoints) {
                this.checkpoints.push({ x: wp.x, y: wp.y, radius: 34, order: baseOrder + order * 0.001 });
                order++;
            }
        }
    }

    _placeScenery() {
        // Buildings + trees are dropped into open space away from roads.
        const bounds = this._bounds();
        const attempts = 260;
        for (let i = 0; i < attempts; i++) {
            const x = randRange(bounds.minX - 150, bounds.maxX + 150);
            const y = randRange(bounds.minY - 150, bounds.maxY + 150);
            if (this._minDistToRoad({ x, y }) < 80) continue;
            if (Math.random() < 0.55) {
                const w = randRange(40, 110), h = randRange(40, 110);
                const rot = Math.random() < 0.5 ? 0 : Math.PI / 2;
                this.buildings.push(this._rectPoly(x, y, w, h, rot));
            } else {
                this.trees.push({ x, y, radius: randRange(10, 20) });
            }
        }
        // Parking areas near a handful of low-degree nodes.
        for (const node of this.nodes) {
            if ((this.degree[node.id] || 0) === 1 && Math.random() < 0.6) {
                this.parking.push(this._rectPoly(node.x + randRange(-60, 60), node.y + randRange(-60, 60), 70, 45, Math.random() * Math.PI));
            }
        }
        // Street lights along roads (used for the night cycle).
        for (const road of this.edges) {
            for (const wp of road.waypoints) {
                if (Math.random() < 0.3) this.streetLights.push({ x: wp.x + road.normal.x * (road.width / 2 + 14), y: wp.y + road.normal.y * (road.width / 2 + 14) });
            }
        }
    }

    _rectPoly(cx, cy, w, h, rot) {
        const hw = w / 2, hh = h / 2;
        const pts = [{ x: -hw, y: -hh }, { x: hw, y: -hh }, { x: hw, y: hh }, { x: -hw, y: hh }];
        return pts.map(p => ({
            x: cx + p.x * Math.cos(rot) - p.y * Math.sin(rot),
            y: cy + p.x * Math.sin(rot) + p.y * Math.cos(rot)
        }));
    }

    _minDistToRoad(p) {
        let min = Infinity;
        for (const road of this.edges) {
            min = Math.min(min, pointToSegmentDist(p, road.a, road.b) - road.width / 2);
        }
        return min;
    }

    _bounds() {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const n of this.nodes) {
            minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
            minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
        }
        return { minX, maxX, minY, maxY };
    }

    // A random directed lane path used to spawn / route both AI and traffic
    // cars: returns an ordered list of {edge, forward} steps starting from
    // a random edge.
    randomPath(steps) {
        let edge = pickRandom(this.edges);
        let forward = Math.random() < 0.5;
        const path = [{ edge, forward }];
        let currentNode = forward ? edge.b : edge.a;
        for (let i = 0; i < steps; i++) {
            const options = this.edges
                .filter(e => (e.a === currentNode || e.b === currentNode) && e !== edge)
                .map(e => ({ edge: e, forward: e.a === currentNode }));
            if (options.length === 0) break;
            const next = pickRandom(options);
            path.push(next);
            edge = next.edge;
            forward = next.forward;
            currentNode = forward ? edge.b : edge.a;
        }
        return path;
    }

    // Builds a single fixed backward path (a list of {edge, forward, fromNode,
    // startCum, entryT} segments) from a starting lane position, long enough
    // to cover `maxDistance`. All population members should walk this SAME
    // path object (see positionAlongQueuePath) - if each car re-picked its
    // own random branch at every intersection independently, different cars
    // could diverge onto different roads and end up overlapping anyway.
    buildQueuePath(startEdge, startForward, maxDistance) {
        let edge = startEdge, forward = startForward;
        let fromNode = forward ? edge.a : edge.b;
        const firstDist = 0.15 * edge.length;
        const segments = [{ edge, forward, fromNode, startCum: 0, entryT: 0.15 }];
        let cumulative = firstDist;
        let guard = 0;
        const recent = [edge]; // avoid ping-ponging back and forth over a short loop of edges
        while (cumulative < maxDistance && guard++ < 1000) {
            const node = fromNode;
            let options = this.edges.filter(e => (e.a === node || e.b === node) && e !== edge);
            const fresh = options.filter(e => !recent.includes(e));
            if (fresh.length > 0) options = fresh;
            if (options.length > 0) {
                edge = pickRandom(options);
                forward = edge.b === node; // arrive at `node` when travelling forward
            } else {
                forward = !forward; // true dead end: double back along the same road
            }
            recent.push(edge);
            if (recent.length > 4) recent.shift();
            fromNode = forward ? edge.a : edge.b;
            segments.push({ edge, forward, fromNode, startCum: cumulative, entryT: 1 });
            cumulative += edge.length;
        }
        return segments;
    }

    // Looks up the world position `distance` units back along a path built
    // by buildQueuePath.
    positionAlongQueuePath(segments, distance) {
        let seg = segments[0];
        for (const s of segments) {
            if (s.startCum <= distance) seg = s; else break;
        }
        const distIntoSeg = distance - seg.startCum;
        const t = Math.max(0, Math.min(1, seg.entryT - distIntoSeg / seg.edge.length));
        return seg.edge.laneCenter(seg.fromNode, t);
    }

    // Walks BACKWARD along the road graph from a starting lane position
    // (edge/forward, at t=0.15 into the edge) by `distance` world-units,
    // hopping through intersections onto a connected edge whenever it runs
    // off the current one. Convenience wrapper for a single lookup; for a
    // whole population use buildQueuePath + positionAlongQueuePath instead
    // so every member walks the same fixed path (see above).
    queuePosition(startEdge, startForward, distance) {
        const segments = this.buildQueuePath(startEdge, startForward, distance);
        return this.positionAlongQueuePath(segments, distance);
    }
}

class Road {
    constructor(a, b, width, lanesEachWay, isHighway) {
        this.a = a; this.b = b;
        this.width = width;
        this.lanesEachWay = lanesEachWay;
        this.isHighway = isHighway;
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy);
        this.dir = { x: dx / len, y: dy / len };
        this.normal = { x: -this.dir.y, y: this.dir.x };
        this.length = len;
    }

    build(radiusA, radiusB) {
        // Trim the drivable boundary back from the intersection nodes so
        // cars have room to turn / cross without hitting a wall.
        const startA = { x: this.a.x + this.dir.x * radiusA, y: this.a.y + this.dir.y * radiusA };
        const endB = { x: this.b.x - this.dir.x * radiusB, y: this.b.y - this.dir.y * radiusB };
        const hw = this.width / 2;
        this.leftBorder = {
            a: { x: startA.x + this.normal.x * hw, y: startA.y + this.normal.y * hw },
            b: { x: endB.x + this.normal.x * hw, y: endB.y + this.normal.y * hw }
        };
        this.rightBorder = {
            a: { x: startA.x - this.normal.x * hw, y: startA.y - this.normal.y * hw },
            b: { x: endB.x - this.normal.x * hw, y: endB.y - this.normal.y * hw }
        };

        // Waypoints (checkpoints + traffic navigation) spaced along the road.
        this.waypoints = [];
        const count = Math.max(1, Math.floor(this.length / 220));
        for (let i = 1; i <= count; i++) {
            const t = i / (count + 1);
            this.waypoints.push({ x: lerp(this.a.x, this.b.x, t), y: lerp(this.a.y, this.b.y, t) });
        }

        // Stop-line positions near each end, offset into the "forward" lane.
        this.stopLineA = this._stopLine(this.a, radiusA, 1); // approach travelling a->? actually entering node a
        this.stopLineB = this._stopLine(this.b, radiusB, -1);
    }

    _stopLine(node, radius, sign) {
        const base = { x: node.x - this.dir.x * radius * sign, y: node.y - this.dir.y * radius * sign };
        // offset into the right-hand lane (driving on the right)
        const laneOffset = this.width / 4;
        const off = { x: this.normal.x * laneOffset * sign, y: this.normal.y * laneOffset * sign };
        return { x: base.x + off.x, y: base.y + off.y };
    }

    // World-space point offset to the right-hand lane centre, travelling
    // from `from` node towards `to` node at parametric position t (0..1).
    laneCenter(fromNode, t) {
        const forward = fromNode === this.a;
        const start = forward ? this.a : this.b;
        const end = forward ? this.b : this.a;
        const px = lerp(start.x, end.x, t);
        const py = lerp(start.y, end.y, t);
        const dirSign = forward ? 1 : -1;
        const laneOffset = this.width / 4;
        // The "right hand side" relative to travel direction:
        const rightNormalSign = forward ? 1 : -1;
        const travelAngle = Math.atan2(end.y - start.y, end.x - start.x);
        return {
            x: px + this.normal.x * laneOffset * rightNormalSign,
            y: py + this.normal.y * laneOffset * rightNormalSign,
            // Converted into this project's car-angle convention, where
            // angle 0 means "facing up" and movement uses
            // (x -= sin(angle)*speed, y -= cos(angle)*speed).
            angle: travelAngle - Math.PI / 2
        };
    }
}

/* ========================== 3. TRAFFIC LIGHTS =========================== */

class TrafficLight {
    static GREEN = 'GREEN'; static YELLOW = 'YELLOW'; static RED = 'RED';

    constructor(node, connectedRoads) {
        this.node = node;
        this.roads = connectedRoads;
        // Split approaches into two phase groups by their approximate
        // heading (near-horizontal vs near-vertical / diagonal).
        this.phaseOf = new Map();
        for (const road of connectedRoads) {
            const angle = Math.atan2(road.dir.y, road.dir.x);
            const norm = ((angle % Math.PI) + Math.PI) % Math.PI; // 0..PI
            const group = (norm > Math.PI / 4 && norm < 3 * Math.PI / 4) ? 1 : 0;
            this.phaseOf.set(road, group);
        }
        this.timer = randRange(0, 6);
        this.cycle = ['A_GREEN', 'A_YELLOW', 'ALL_RED_1', 'B_GREEN', 'B_YELLOW', 'ALL_RED_2'];
        this.durations = { A_GREEN: 7, A_YELLOW: 1.6, ALL_RED_1: 0.6, B_GREEN: 7, B_YELLOW: 1.6, ALL_RED_2: 0.6 };
        this.stateIndex = 0;
    }

    update(dt) {
        this.timer += dt;
        const stateName = this.cycle[this.stateIndex];
        if (this.timer >= this.durations[stateName]) {
            this.timer = 0;
            this.stateIndex = (this.stateIndex + 1) % this.cycle.length;
        }
    }

    stateFor(road) {
        const group = this.phaseOf.get(road);
        const stateName = this.cycle[this.stateIndex];
        if (stateName.startsWith('ALL_RED')) return TrafficLight.RED;
        const activeGroup = stateName.startsWith('A') ? 0 : 1;
        if (group !== activeGroup) return TrafficLight.RED;
        if (stateName.endsWith('YELLOW')) return TrafficLight.YELLOW;
        return TrafficLight.GREEN;
    }

    // Returns the world position of the stop-line for a given approach road.
    stopPointFor(road) {
        return road.a === this.node ? road.stopLineA : road.stopLineB;
    }
}

/* =============================== 4. SENSOR =============================== */

class Sensor {
    constructor(car, rayCount = 7, rayLength = 180, raySpread = Math.PI * 0.8) {
        this.car = car;
        this.rayCount = rayCount;
        this.rayLength = rayLength;
        this.raySpread = raySpread;
        this.rays = [];
        this.readings = [];
    }

    update(borders, traffic) {
        this.rays = [];
        for (let i = 0; i < this.rayCount; i++) {
            const rayAngle = lerp(
                this.raySpread / 2, -this.raySpread / 2,
                this.rayCount === 1 ? 0.5 : i / (this.rayCount - 1)
            ) + this.car.angle;
            const start = { x: this.car.x, y: this.car.y };
            const end = {
                x: this.car.x - Math.sin(rayAngle) * this.rayLength,
                y: this.car.y - Math.cos(rayAngle) * this.rayLength
            };
            this.rays.push([start, end]);
        }
        this.readings = this.rays.map(ray => this._getReading(ray, borders, traffic));
    }

    _getReading(ray, borders, traffic) {
        let touches = [];
        for (const border of borders) {
            const touch = getIntersection(ray[0], ray[1], border.a, border.b);
            if (touch) touches.push(touch);
        }
        for (const other of traffic) {
            if (other === this.car) continue;
            const poly = other.polygon;
            for (let i = 0; i < poly.length; i++) {
                const touch = getIntersection(ray[0], ray[1], poly[i], poly[(i + 1) % poly.length]);
                if (touch) touches.push(touch);
            }
        }
        if (touches.length === 0) return null;
        const offsets = touches.map(t => t.offset);
        const minOffset = Math.min(...offsets);
        return touches.find(t => t.offset === minOffset);
    }

    // Normalised inputs (0 = clear, ~1 = something touching the car).
    getNormalizedInputs() {
        return this.readings.map(r => r === null ? 0 : 1 - r.offset);
    }
}

/* ========================= 5. NEURAL NETWORK =========================== */

class Layer {
    constructor(numIn, numOut) {
        this.numIn = numIn; this.numOut = numOut;
        this.weights = [];
        this.biases = new Array(numOut).fill(0).map(() => randRange(-1, 1));
        for (let i = 0; i < numIn; i++) {
            this.weights.push(new Array(numOut).fill(0).map(() => randRange(-1, 1)));
        }
    }

    clone() {
        const l = new Layer(this.numIn, this.numOut);
        l.biases = this.biases.slice();
        l.weights = this.weights.map(row => row.slice());
        return l;
    }

    static forward(inputs, layer) {
        const rawOutputs = new Array(layer.numOut).fill(0);
        for (let o = 0; o < layer.numOut; o++) {
            let sum = layer.biases[o];
            for (let i = 0; i < layer.numIn; i++) sum += inputs[i] * layer.weights[i][o];
            rawOutputs[o] = sum;
        }
        return rawOutputs;
    }
}

function tanhAct(x) { return Math.tanh(x); }
function sigmoidAct(x) { return 1 / (1 + Math.exp(-x)); }

class NeuralNetwork {
    // sizes e.g. [numSensors+3, 10, 8, 3]
    constructor(sizes) {
        this.sizes = sizes;
        this.layers = [];
        for (let i = 0; i < sizes.length - 1; i++) {
            this.layers.push(new Layer(sizes[i], sizes[i + 1]));
        }
    }

    clone() {
        const n = new NeuralNetwork(this.sizes);
        n.layers = this.layers.map(l => l.clone());
        return n;
    }

    toJSON() {
        return { sizes: this.sizes, layers: this.layers.map(l => ({ weights: l.weights, biases: l.biases })) };
    }

    static fromJSON(json) {
        const n = new NeuralNetwork(json.sizes);
        n.layers.forEach((l, i) => { l.weights = json.layers[i].weights; l.biases = json.layers[i].biases; });
        return n;
    }

    // Returns [steer(-1..1), throttle(0..1), brake(0..1)] plus raw
    // per-layer activations (for the visualiser).
    static feedForward(inputs, network) {
        let outputs = inputs;
        const activations = [inputs];
        for (let i = 0; i < network.layers.length; i++) {
            const raw = Layer.forward(outputs, network.layers[i]);
            const isOutputLayer = i === network.layers.length - 1;
            if (isOutputLayer) {
                outputs = [tanhAct(raw[0]), sigmoidAct(raw[1]), sigmoidAct(raw[2])];
            } else {
                outputs = raw.map(tanhAct);
            }
            activations.push(outputs);
        }
        return { outputs, activations };
    }

    static mutate(network, amount) {
        for (const layer of network.layers) {
            for (let i = 0; i < layer.biases.length; i++) {
                layer.biases[i] = lerp(layer.biases[i], randRange(-1, 1), amount);
            }
            for (let i = 0; i < layer.weights.length; i++) {
                for (let j = 0; j < layer.weights[i].length; j++) {
                    layer.weights[i][j] = lerp(layer.weights[i][j], randRange(-1, 1), amount);
                }
            }
        }
    }
}

/* ================================ 6. CAR ================================ */

let nextCarId = 1;

class Car {
    constructor(x, y, angle, type, network = null) {
        this.id = nextCarId++;
        this.x = x; this.y = y; this.angle = angle;
        this.type = type; // 'AI' or 'TRAFFIC'
        this.width = 26; this.height = 46;
        this.speed = 0;
        this.acceleration = 0.24;
        this.maxSpeed = type === 'AI' ? 3.6 : randRange(1.6, 2.4);
        this.friction = 0.045;
        this.steerSpeed = 0.045;
        this.damaged = false;
        this.color = type === 'AI'
            ? `hsl(${randRange(180, 260)}, 70%, ${randRange(45, 65)}%)`
            : `hsl(${randRange(0, 360)}, 55%, 50%)`;

        if (type === 'AI') {
            this.sensor = new Sensor(this);
            this.brain = network || new NeuralNetwork([this.sensor.rayCount + 3, 10, 8, 3]);
        }

        // Fitness / stats
        this.distanceTraveled = 0;
        this.checkpointScore = 0;
        this.visitedCheckpoints = new Set();
        this.timeAlive = 0;
        this.stuckTimer = 0;
        this.generation = 0;
        this.currentCheckpointCount = 0;

        // Traffic-car routing
        if (type === 'TRAFFIC') {
            this.path = null;
            this.pathIndex = 0;
            this.progress = 0; // 0..1 along current edge
        }
    }

    get polygon() {
        const points = [];
        const rad = Math.hypot(this.width, this.height) / 2;
        const alpha = Math.atan2(this.width, this.height);
        points.push({ x: this.x - Math.sin(this.angle - alpha) * rad, y: this.y - Math.cos(this.angle - alpha) * rad });
        points.push({ x: this.x - Math.sin(this.angle + alpha) * rad, y: this.y - Math.cos(this.angle + alpha) * rad });
        points.push({ x: this.x - Math.sin(Math.PI + this.angle - alpha) * rad, y: this.y - Math.cos(Math.PI + this.angle - alpha) * rad });
        points.push({ x: this.x - Math.sin(Math.PI + this.angle + alpha) * rad, y: this.y - Math.cos(Math.PI + this.angle + alpha) * rad });
        return points;
    }

    update(dt, network, otherCars, buildings) {
        if (this.damaged) return;
        this.timeAlive += dt;

        if (this.type === 'AI') {
            this.sensor.update(network.borders, otherCars);
            const inputs = [
                ...this.sensor.getNormalizedInputs(),
                this.speed / this.maxSpeed,
                Math.sin(this.angle),
                Math.cos(this.angle)
            ];
            const { outputs, activations } = NeuralNetwork.feedForward(inputs, this.brain);
            this.lastActivations = activations;
            const [steer, throttle, brake] = outputs;
            this._drive(steer, throttle, brake, dt);
        } else {
            this._driveTraffic(network, dt);
        }

        this._move(dt);
        this._assessDamage(network, otherCars, buildings);
        this._trackFitness(network);
    }

    _drive(steer, throttle, brake, dt) {
        if (throttle > 0.15) this.speed += this.acceleration * throttle;
        if (brake > 0.15) this.speed -= this.acceleration * 1.6 * brake;
        this.speed *= (1 - this.friction);
        this.speed = Math.max(-this.maxSpeed / 2, Math.min(this.maxSpeed, this.speed));
        const speedFactor = Math.min(1, Math.abs(this.speed) / (this.maxSpeed * 0.4) + 0.15);
        const flip = this.speed >= 0 ? 1 : -1;
        this.angle -= steer * this.steerSpeed * speedFactor * flip;
    }

    _driveTraffic(network, dt) {
        if (!this.path) this._pickNewPath(network);
        const step = this.path[this.pathIndex];
        if (!step) { this._pickNewPath(network); return; }
        const road = step.edge;
        const fromNode = step.forward ? road.a : road.b;
        this.progress += this.speed / Math.max(road.length, 1);

        const target = road.laneCenter(fromNode, Math.min(1, this.progress));
        const desiredAngle = target.angle;
        let diff = desiredAngle - this.angle;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        this.angle += Math.max(-this.steerSpeed * 1.2, Math.min(this.steerSpeed * 1.2, diff));
        // Nudge position toward the lane centreline (keeps traffic orderly).
        this.x = lerp(this.x, target.x, 0.12);
        this.y = lerp(this.y, target.y, 0.12);

        let desiredSpeed = this.maxSpeed;

        // Slow / stop for a car ahead in the same lane.
        for (const other of network.__trafficCars || []) {
            if (other === this) continue;
            const d = dist(this, other);
            if (d < 70) {
                const ahead = Math.cos(this.angle - angleOf(this, other));
                if (d < 55 && ahead > 0.5) desiredSpeed = 0;
                else if (d < 70 && ahead > 0.3) desiredSpeed = Math.min(desiredSpeed, other.speed * 0.9);
            }
        }
        // Also react to nearby AI cars.
        for (const other of network.__aiCars || []) {
            const d = dist(this, other);
            if (d < 60 && Math.cos(this.angle - angleOf(this, other)) > 0.5) desiredSpeed = 0;
        }

        // Traffic-light awareness near the end of this edge.
        if (this.progress > 0.75) {
            const endNode = step.forward ? road.b : road.a;
            const light = network.trafficLights.find(l => l.node === endNode);
            if (light) {
                const state = light.stateFor(road);
                if (state !== TrafficLight.GREEN && this.progress < 0.98) desiredSpeed = 0;
            }
        }

        if (this.speed < desiredSpeed) this.speed += this.acceleration * 0.6;
        else this.speed -= this.acceleration * 0.9;
        this.speed = Math.max(0, Math.min(this.maxSpeed, this.speed));

        if (this.progress >= 1) {
            this.pathIndex++;
            this.progress = 0;
            if (this.pathIndex >= this.path.length - 1) this._pickNewPath(network);
        }
    }

    _pickNewPath(network) {
        this.path = network.randomPath(12);
        this.pathIndex = 0;
        this.progress = 0;
        const step = this.path[0];
        const fromNode = step.forward ? step.edge.a : step.edge.b;
        const c = step.edge.laneCenter(fromNode, 0.02);
        this.x = c.x; this.y = c.y; this.angle = c.angle;
    }

    _move(dt) {
        const factor = dt * 60;
        this.x -= Math.sin(this.angle) * this.speed * factor;
        this.y -= Math.cos(this.angle) * this.speed * factor;
        this.distanceTraveled += Math.abs(this.speed) * factor;
        if (Math.abs(this.speed) < 0.02) this.stuckTimer += dt; else this.stuckTimer = 0;
        if (this.stuckTimer > 4 && this.type === 'AI') this.damaged = true; // gave up / stuck
    }

    _assessDamage(network, otherCars, buildings) {
        if (this.type !== 'AI') return;
        const poly = this.polygon;
        for (const border of network.borders) {
            if (getIntersection(poly[0], poly[1], border.a, border.b) ||
                getIntersection(poly[1], poly[2], border.a, border.b) ||
                getIntersection(poly[2], poly[3], border.a, border.b) ||
                getIntersection(poly[3], poly[0], border.a, border.b)) {
                this.damaged = true; return;
            }
        }
        for (const other of otherCars) {
            if (other === this) continue;
            if (polysIntersect(poly, other.polygon)) { this.damaged = true; return; }
        }
        for (const b of buildings) {
            if (polysIntersect(poly, b)) { this.damaged = true; return; }
        }
    }

    _trackFitness(network) {
        if (this.type !== 'AI') return;
        for (let i = 0; i < network.checkpoints.length; i++) {
            if (this.visitedCheckpoints.has(i)) continue;
            const cp = network.checkpoints[i];
            if (dist(this, cp) < cp.radius) {
                this.visitedCheckpoints.add(i);
                this.checkpointScore += 12 + cp.order * 3;
                this.currentCheckpointCount++;
            }
        }
    }

    get fitness() {
        if (this.type !== 'AI') return 0;
        const crashPenalty = this.damaged && this.stuckTimer <= 4 ? 40 : 0;
        return this.distanceTraveled * 0.05 + this.checkpointScore - crashPenalty;
    }

    draw(ctx, drawSensor, isSelected, night) {
        const poly = this.polygon;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(poly[0].x, poly[0].y);
        for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
        ctx.closePath();
        ctx.fillStyle = this.damaged ? '#4b4b4b' : this.color;
        ctx.shadowColor = 'rgba(0,0,0,0.35)';
        ctx.shadowBlur = 6; ctx.shadowOffsetY = 2;
        ctx.fill();
        ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
        ctx.strokeStyle = isSelected ? '#ffdd33' : 'rgba(0,0,0,0.35)';
        ctx.lineWidth = isSelected ? 3 : 1;
        ctx.stroke();

        // windshield
        ctx.fillStyle = 'rgba(20,30,45,0.7)';
        const midFront = { x: (poly[0].x + poly[1].x) / 2, y: (poly[0].y + poly[1].y) / 2 };
        const midMid = { x: (poly[0].x + poly[1].x + poly[2].x + poly[3].x) / 4, y: (poly[0].y + poly[1].y + poly[2].y + poly[3].y) / 4 };
        ctx.beginPath();
        ctx.arc((midFront.x + midMid.x) / 2, (midFront.y + midMid.y) / 2, this.width * 0.32, 0, Math.PI * 2);
        ctx.fill();

        if (night && !this.damaged) {
            ctx.fillStyle = 'rgba(255,240,160,0.9)';
            ctx.beginPath(); ctx.arc(poly[0].x, poly[0].y, 3, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(poly[1].x, poly[1].y, 3, 0, Math.PI * 2); ctx.fill();
        }
        ctx.restore();

        if (drawSensor && this.sensor) {
            for (let i = 0; i < this.sensor.rays.length; i++) {
                const ray = this.sensor.rays[i];
                const reading = this.sensor.readings[i];
                const end = reading || ray[1];
                ctx.beginPath();
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(255,220,0,0.75)';
                ctx.moveTo(ray[0].x, ray[0].y); ctx.lineTo(end.x, end.y);
                ctx.stroke();
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(0,0,0,0.55)';
                ctx.moveTo(ray[1].x, ray[1].y); ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }
        }
    }
}

/* ========================= 7. EVOLUTION MANAGER ========================= */

class EvolutionManager {
    constructor(network, populationSize) {
        this.network = network;
        this.populationSize = populationSize;
        this.generation = 1;
        this.bestEverBrain = null;
        this.bestEverFitness = 0;
        this.genTimer = 0;
        this.maxGenTime = 40; // seconds of sim-time per generation
        this.cars = [];
        this._spawnPopulation();
    }

    _spawnEdgeAndAngle() {
        // Prefer a well-connected starting edge - this keeps the backward
        // spawn queue (see RoadNetwork.queuePosition) from hitting a dead
        // end after only one or two hops.
        const wellConnected = this.network.edges.filter(e =>
            (this.network.degree[e.a.id] || 0) >= 3 || (this.network.degree[e.b.id] || 0) >= 3);
        const road = pickRandom(wellConnected.length ? wellConnected : this.network.edges);
        const forward = Math.random() < 0.5;
        const from = forward ? road.a : road.b;
        const c = road.laneCenter(from, 0.15);
        return { x: c.x, y: c.y, angle: c.angle, edge: road, forward };
    }

    _spawnPopulation(seedBrain = null) {
        this.cars = [];
        const spacing = 85; // clears the car body (46 long) with margin for corners
        const minSafeCenterDist = 70; // > car diagonal (~53) with margin - guarantees no polygon overlap
        // A city this size is too small for one giant nose-to-tail queue (or
        // even several) to guarantee no self-intersection - the graph can
        // loop back on itself. So on top of walking the road graph, actively
        // check each candidate spot against every car already placed and
        // keep stepping forward (or start a fresh chain) until it's clear.
        let spawn = this._spawnEdgeAndAngle();
        let chainPath = this.network.buildQueuePath(spawn.edge, spawn.forward, spacing * (this.populationSize + 8));
        let chainDist = 0;
        for (let i = 0; i < this.populationSize; i++) {
            let brain;
            const frac = i / this.populationSize;
            if (seedBrain && frac < 0.85) {
                brain = seedBrain.clone();
                // Tiered mutation strength keeps the elite intact while the
                // rest of the population explores nearby (and further away)
                // variations, so a plateau can still be escaped over time.
                if (frac > 0) {
                    const amount = frac < 0.15 ? 0.06 : frac < 0.4 ? 0.18 : frac < 0.7 ? 0.35 : 0.6;
                    NeuralNetwork.mutate(brain, amount);
                }
            } else {
                brain = null; // fresh random genome - keeps genetic diversity up
            }

            let pos, attempts = 0;
            while (true) {
                chainDist += spacing;
                pos = this.network.positionAlongQueuePath(chainPath, chainDist);
                const collides = this.cars.some(c => dist(c, pos) < minSafeCenterDist);
                attempts++;
                if (!collides || attempts > 80) break;
            }
            if (attempts > 80) {
                // This chain wound up somewhere too tangled to escape -
                // start a brand new one from a fresh random spot instead.
                spawn = this._spawnEdgeAndAngle();
                chainPath = this.network.buildQueuePath(spawn.edge, spawn.forward, spacing * (this.populationSize + 8));
                chainDist = 0;
                pos = this.network.positionAlongQueuePath(chainPath, 0);
            }

            const car = new Car(
                pos.x + randRange(-2, 2),
                pos.y + randRange(-2, 2),
                pos.angle, 'AI', brain
            );
            car.generation = this.generation;
            this.cars.push(car);
        }
        this.genTimer = 0;
    }

    update(dt, traffic, buildings) {
        this.genTimer += dt;
        const allCars = [...this.cars, ...traffic];
        for (const car of this.cars) car.update(dt, this.network, allCars, buildings);

        const alive = this.cars.filter(c => !c.damaged);
        if (alive.length === 0 || this.genTimer > this.maxGenTime) {
            this.nextGeneration();
        }
    }

    nextGeneration() {
        const ranked = this.cars.slice().sort((a, b) => b.fitness - a.fitness);
        const best = ranked[0];
        if (best && best.fitness > this.bestEverFitness) {
            this.bestEverFitness = best.fitness;
            this.bestEverBrain = best.brain.clone();
        }
        this.generation++;
        this._spawnPopulation(this.bestEverBrain || (best ? best.brain : null));
    }

    get aliveCount() { return this.cars.filter(c => !c.damaged).length; }
    get bestCar() {
        return this.cars.reduce((best, c) => (!best || c.fitness > best.fitness) ? c : best, null);
    }
}

/* ================================ 8. CAMERA ================================ */

class Camera {
    constructor(canvas) {
        this.canvas = canvas;
        this.x = 0; this.y = 0; // world point at screen centre
        this.zoom = 1;
        this.minZoom = 0.25; this.maxZoom = 2.5;
        this.dragging = false;
        this.dragStart = null;
        this.followCar = null;
        this._wire();
    }

    _wire() {
        const c = this.canvas;
        c.addEventListener('mousedown', e => {
            this.dragging = true;
            this.followCar = null;
            this.dragStart = { x: e.clientX, y: e.clientY, camX: this.x, camY: this.y };
        });
        window.addEventListener('mouseup', () => this.dragging = false);
        window.addEventListener('mousemove', e => {
            if (!this.dragging) return;
            const dx = (e.clientX - this.dragStart.x) / this.zoom;
            const dy = (e.clientY - this.dragStart.y) / this.zoom;
            this.x = this.dragStart.camX - dx;
            this.y = this.dragStart.camY - dy;
        });
        c.addEventListener('wheel', e => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.1 : 0.9;
            this.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoom * factor));
        }, { passive: false });
    }

    reset(worldCenter) {
        this.x = worldCenter.x; this.y = worldCenter.y; this.zoom = 0.85;
        this.followCar = null;
    }

    update() {
        if (this.followCar && !this.followCar.damaged) {
            this.x = lerp(this.x, this.followCar.x, 0.08);
            this.y = lerp(this.y, this.followCar.y, 0.08);
        }
    }

    applyTransform(ctx) {
        ctx.translate(this.canvas.width / 2, this.canvas.height / 2);
        ctx.scale(this.zoom, this.zoom);
        ctx.translate(-this.x, -this.y);
    }

    screenToWorld(sx, sy) {
        return {
            x: (sx - this.canvas.width / 2) / this.zoom + this.x,
            y: (sy - this.canvas.height / 2) / this.zoom + this.y
        };
    }
}

/* =============================== 9. RENDERER ============================== */

const Renderer = {
    drawWorld(ctx, network, night) {
        this._drawGround(ctx, network, night);
        this._drawParking(ctx, network);
        this._drawSidewalks(ctx, network);
        this._drawRoads(ctx, network, night);
        this._drawStopLinesAndLights(ctx, network);
        this._drawBuildings(ctx, network, night);
        this._drawTrees(ctx, network);
        if (night) this._drawStreetLights(ctx, network);
    },

    _drawGround(ctx, network, night) {
        const b = network._bounds();
        ctx.fillStyle = night ? '#16241a' : '#7fb86b';
        ctx.fillRect(b.minX - 600, b.minY - 600, (b.maxX - b.minX) + 1200, (b.maxY - b.minY) + 1200);
    },

    _drawParking(ctx, network) {
        ctx.fillStyle = 'rgba(90,90,100,0.85)';
        for (const p of network.parking) {
            ctx.beginPath();
            ctx.moveTo(p[0].x, p[0].y);
            for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
            ctx.closePath(); ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1.5;
            for (let i = -1; i <= 1; i += 2) {
                ctx.beginPath();
                ctx.moveTo((p[0].x + p[1].x) / 2 + i * 8, (p[0].y + p[1].y) / 2);
                ctx.lineTo((p[3].x + p[2].x) / 2 + i * 8, (p[3].y + p[2].y) / 2);
                ctx.stroke();
            }
        }
    },

    _drawSidewalks(ctx, network) {
        ctx.strokeStyle = '#c9c4b8';
        for (const road of network.edges) {
            ctx.lineWidth = road.width + 16;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(road.a.x, road.a.y);
            ctx.lineTo(road.b.x, road.b.y);
            ctx.stroke();
        }
    },

    _drawRoads(ctx, network, night) {
        for (const road of network.edges) {
            ctx.strokeStyle = night ? '#26282e' : '#3a3b40';
            ctx.lineWidth = road.width;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(road.a.x, road.a.y);
            ctx.lineTo(road.b.x, road.b.y);
            ctx.stroke();
        }
        // Intersection fill so corners look smooth.
        for (const node of network.nodes) {
            const r = network.nodeRadius[node.id];
            const connected = network.edges.filter(e => e.a === node || e.b === node);
            if (connected.length < 2) continue;
            ctx.fillStyle = night ? '#26282e' : '#3a3b40';
            ctx.beginPath(); ctx.arc(node.x, node.y, r, 0, Math.PI * 2); ctx.fill();
        }
        // Lane markings.
        for (const road of network.edges) {
            ctx.save();
            ctx.strokeStyle = 'rgba(255,255,255,0.85)';
            if (road.lanesEachWay > 1) {
                ctx.setLineDash([16, 14]);
                ctx.lineWidth = 2.5;
                ctx.beginPath(); ctx.moveTo(road.a.x, road.a.y); ctx.lineTo(road.b.x, road.b.y); ctx.stroke();
            }
            ctx.setLineDash([]);
            ctx.strokeStyle = 'rgba(255,220,60,0.9)';
            ctx.lineWidth = 2.5;
            ctx.beginPath(); ctx.moveTo(road.a.x, road.a.y); ctx.lineTo(road.b.x, road.b.y); ctx.stroke();
            ctx.restore();
        }
    },

    _drawStopLinesAndLights(ctx, network) {
        for (const light of network.trafficLights) {
            for (const road of light.roads) {
                const stop = light.stopPointFor(road);
                const state = light.stateFor(road);
                ctx.save();
                ctx.translate(stop.x, stop.y);
                ctx.rotate(Math.atan2(road.dir.y, road.dir.x) + Math.PI / 2);
                ctx.fillStyle = 'rgba(255,255,255,0.9)';
                ctx.fillRect(-road.width / 4 - 3, -3, road.width / 2 + 6, 6);
                ctx.restore();

                ctx.beginPath();
                ctx.fillStyle = state === TrafficLight.GREEN ? '#33e07a' : state === TrafficLight.YELLOW ? '#ffd23f' : '#ff4d4d';
                ctx.arc(stop.x - road.normal.x * (road.width / 2 + 20), stop.y - road.normal.y * (road.width / 2 + 20), 6, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    },

    _drawBuildings(ctx, network, night) {
        for (const poly of network.buildings) {
            ctx.beginPath();
            ctx.moveTo(poly[0].x, poly[0].y);
            for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
            ctx.closePath();
            ctx.fillStyle = night ? '#22262f' : '#9aa0ad';
            ctx.fill();
            ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1.5; ctx.stroke();
            if (night) {
                const cx = (poly[0].x + poly[2].x) / 2, cy = (poly[0].y + poly[2].y) / 2;
                for (let i = 0; i < 3; i++) {
                    if (Math.random() < 0.6) {
                        ctx.fillStyle = 'rgba(255,225,140,0.8)';
                        ctx.fillRect(cx - 8 + i * 8, cy - 6, 4, 4);
                    }
                }
            }
        }
    },

    _drawTrees(ctx, network) {
        for (const t of network.trees) {
            ctx.beginPath();
            ctx.fillStyle = '#3f7a3a';
            ctx.arc(t.x, t.y, t.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#5e4530';
            ctx.fillRect(t.x - 2, t.y + t.radius - 3, 4, 8);
        }
    },

    _drawStreetLights(ctx, network) {
        for (const s of network.streetLights) {
            const grad = ctx.createRadialGradient(s.x, s.y, 2, s.x, s.y, 45);
            grad.addColorStop(0, 'rgba(255,235,160,0.35)');
            grad.addColorStop(1, 'rgba(255,235,160,0)');
            ctx.fillStyle = grad;
            ctx.beginPath(); ctx.arc(s.x, s.y, 45, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#ffe9a3';
            ctx.beginPath(); ctx.arc(s.x, s.y, 3, 0, Math.PI * 2); ctx.fill();
        }
    },

    drawCheckpoints(ctx, network, visitedSet) {
        for (let i = 0; i < network.checkpoints.length; i++) {
            const cp = network.checkpoints[i];
            ctx.beginPath();
            ctx.strokeStyle = visitedSet && visitedSet.has(i) ? 'rgba(60,220,120,0.85)' : 'rgba(255,255,255,0.35)';
            ctx.lineWidth = 2;
            ctx.arc(cp.x, cp.y, cp.radius, 0, Math.PI * 2);
            ctx.stroke();
        }
    },

    drawBorders(ctx, network) {
        ctx.strokeStyle = 'rgba(255,60,60,0.9)';
        ctx.lineWidth = 3;
        for (const b of network.borders) {
            ctx.beginPath(); ctx.moveTo(b.a.x, b.a.y); ctx.lineTo(b.b.x, b.b.y); ctx.stroke();
        }
    },

    drawMinimap(ctx, canvas, network, aiCars, trafficCars, selected, camera) {
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(15,18,24,0.85)';
        ctx.fillRect(0, 0, w, h);
        const b = network._bounds();
        const pad = 200;
        const worldW = (b.maxX - b.minX) + pad * 2, worldH = (b.maxY - b.minY) + pad * 2;
        const scale = Math.min(w / worldW, h / worldH);
        const ox = -( b.minX - pad) * scale, oy = -(b.minY - pad) * scale;

        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 1;
        for (const road of network.edges) {
            ctx.beginPath();
            ctx.moveTo(road.a.x * scale + ox, road.a.y * scale + oy);
            ctx.lineTo(road.b.x * scale + ox, road.b.y * scale + oy);
            ctx.stroke();
        }
        ctx.fillStyle = '#7dd3fc';
        for (const c of trafficCars) ctx.fillRect(c.x * scale + ox - 1, c.y * scale + oy - 1, 2, 2);
        ctx.fillStyle = '#4ade80';
        for (const c of aiCars) {
            if (c.damaged) continue;
            ctx.fillRect(c.x * scale + ox - 1.5, c.y * scale + oy - 1.5, 3, 3);
        }
        if (selected) {
            ctx.fillStyle = '#facc15';
            ctx.beginPath(); ctx.arc(selected.x * scale + ox, selected.y * scale + oy, 4, 0, Math.PI * 2); ctx.fill();
        }
        // camera viewport rectangle
        const vw = (camera.canvas.width / camera.zoom) * scale;
        const vh = (camera.canvas.height / camera.zoom) * scale;
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(camera.x * scale + ox - vw / 2, camera.y * scale + oy - vh / 2, vw, vh);
    }
};

/* ============================ 10. UI + MAIN LOOP =========================== */

(function main() {
    const canvas = document.getElementById('worldCanvas');
    const ctx = canvas.getContext('2d');
    const minimapCanvas = document.getElementById('minimapCanvas');
    const minimapCtx = minimapCanvas.getContext('2d');
    const nnCanvas = document.getElementById('nnCanvas');
    const nnCtx = nnCanvas.getContext('2d');

    function resize() {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
    }
    window.addEventListener('resize', resize);

    const network = new RoadNetwork();
    const camera = new Camera(canvas);
    resize();
    const b = network._bounds();
    camera.reset({ x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2 });

    // ---- state ----
    const state = {
        running: false,
        speedMultiplier: 1,
        showSensors: true,
        showNN: true,
        showCheckpoints: false,
        showDebug: true,
        showBorders: false,
        night: false,
        selectedCar: null
    };

    let evo = new EvolutionManager(network, 60);
    const trafficCars = [];
    for (let i = 0; i < 26; i++) {
        const car = new Car(0, 0, 0, 'TRAFFIC');
        car._pickNewPath(network);
        trafficCars.push(car);
    }
    network.__trafficCars = trafficCars;
    network.__aiCars = evo.cars;

    // ---- UI elements ----
    const el = id => document.getElementById(id);
    const btnStart = el('btnStart'), btnPause = el('btnPause'), btnNextGen = el('btnNextGen'), btnReset = el('btnReset');
    const carsInput = el('carsInput');
    const speedButtons = document.querySelectorAll('.speed-btn');
    const cbSensors = el('cbSensors'), cbNN = el('cbNN'), cbCheckpoints = el('cbCheckpoints'), cbDebug = el('cbDebug'), cbBorders = el('cbBorders'), cbNight = el('cbNight');
    const statGen = el('statGen'), statAlive = el('statAlive'), statBestDist = el('statBestDist'), statBestFit = el('statBestFit'), statAvgFit = el('statAvgFit'), statFps = el('statFps');
    const carInfoPanel = el('carInfoPanel');
    const btnResetCam = el('btnResetCam');
    const btnSave = el('btnSave'), btnLoad = el('btnLoad'), btnResetTraining = el('btnResetTraining');

    btnStart.addEventListener('click', () => { state.running = true; });
    btnPause.addEventListener('click', () => { state.running = false; });
    btnNextGen.addEventListener('click', () => evo.nextGeneration());
    btnReset.addEventListener('click', () => {
        evo = new EvolutionManager(network, evo.populationSize);
        network.__aiCars = evo.cars;
        state.selectedCar = null;
    });
    carsInput.addEventListener('change', () => {
        const n = Math.max(4, Math.min(300, parseInt(carsInput.value) || 60));
        carsInput.value = n;
        evo = new EvolutionManager(network, n);
        network.__aiCars = evo.cars;
        state.selectedCar = null;
    });
    speedButtons.forEach(btn => btn.addEventListener('click', () => {
        speedButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.speedMultiplier = parseFloat(btn.dataset.speed);
    }));
    cbSensors.addEventListener('change', () => state.showSensors = cbSensors.checked);
    cbNN.addEventListener('change', () => state.showNN = cbNN.checked);
    cbCheckpoints.addEventListener('change', () => state.showCheckpoints = cbCheckpoints.checked);
    cbDebug.addEventListener('change', () => state.showDebug = cbDebug.checked);
    cbBorders.addEventListener('change', () => state.showBorders = cbBorders.checked);
    cbNight.addEventListener('change', () => state.night = cbNight.checked);
    btnResetCam.addEventListener('click', () => camera.reset({ x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2 }));

    btnSave.addEventListener('click', () => {
        const best = evo.bestEverBrain || (evo.bestCar && evo.bestCar.brain);
        if (!best) return;
        const payload = { brain: best.toJSON(), generation: evo.generation, bestFitness: evo.bestEverFitness, populationSize: evo.populationSize };
        localStorage.setItem('drivingSim.save', JSON.stringify(payload));
        flashButton(btnSave, 'Saved!');
    });
    btnLoad.addEventListener('click', () => {
        const raw = localStorage.getItem('drivingSim.save');
        if (!raw) { flashButton(btnLoad, 'No save'); return; }
        const payload = JSON.parse(raw);
        const brain = NeuralNetwork.fromJSON(payload.brain);
        evo = new EvolutionManager(network, payload.populationSize || evo.populationSize);
        evo.generation = payload.generation || 1;
        evo.bestEverFitness = payload.bestFitness || 0;
        evo.bestEverBrain = brain;
        evo._spawnPopulation(brain);
        network.__aiCars = evo.cars;
        carsInput.value = evo.populationSize;
        flashButton(btnLoad, 'Loaded!');
    });
    btnResetTraining.addEventListener('click', () => {
        localStorage.removeItem('drivingSim.save');
        evo = new EvolutionManager(network, evo.populationSize);
        network.__aiCars = evo.cars;
        flashButton(btnResetTraining, 'Cleared');
    });
    function flashButton(btn, text) {
        const old = btn.textContent;
        btn.textContent = text;
        setTimeout(() => btn.textContent = old, 900);
    }

    canvas.addEventListener('click', (e) => {
        if (camera.dragStart && Math.hypot(e.clientX - camera.dragStart.x, e.clientY - camera.dragStart.y) > 6) return;
        const rect = canvas.getBoundingClientRect();
        const world = camera.screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
        let closest = null, closestD = 26;
        for (const c of evo.cars) {
            const d = dist(world, c);
            if (d < closestD) { closest = c; closestD = d; }
        }
        if (closest) { state.selectedCar = closest; camera.followCar = closest; }
    });

    // ---- auto-restore save on load ----
    (function tryAutoLoad() {
        const raw = localStorage.getItem('drivingSim.save');
        if (!raw) return;
        try {
            const payload = JSON.parse(raw);
            const brain = NeuralNetwork.fromJSON(payload.brain);
            evo.bestEverBrain = brain;
            evo.bestEverFitness = payload.bestFitness || 0;
            evo.generation = payload.generation || 1;
        } catch (e) { /* ignore corrupt save */ }
    })();

    // ---- NN visualiser ----
    function drawNetwork(ctx, canvas, car) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!car || !car.brain || !car.lastActivations) return;
        const activations = car.lastActivations;
        const margin = 24;
        const levelCount = activations.length;
        const levelWidth = (canvas.width - margin * 2) / (levelCount - 1);

        function nodeY(levelIndex, nodeIndex, count) {
            return lerp(canvas.height - margin, margin, count === 1 ? 0.5 : nodeIndex / (count - 1));
        }
        function nodeX(levelIndex) { return margin + levelIndex * levelWidth; }

        // connections
        for (let lvl = 0; lvl < levelCount - 1; lvl++) {
            const layer = car.brain.layers[lvl];
            for (let i = 0; i < activations[lvl].length; i++) {
                for (let o = 0; o < activations[lvl + 1].length; o++) {
                    const w = layer.weights[i][o];
                    ctx.beginPath();
                    ctx.strokeStyle = getRGBA(Math.max(-1, Math.min(1, w)));
                    ctx.lineWidth = 1.4;
                    ctx.moveTo(nodeX(lvl), nodeY(lvl, i, activations[lvl].length));
                    ctx.lineTo(nodeX(lvl + 1), nodeY(lvl + 1, o, activations[lvl + 1].length));
                    ctx.stroke();
                }
            }
        }
        // nodes
        const labelsOut = ['STEER', 'GAS', 'BRAKE'];
        for (let lvl = 0; lvl < levelCount; lvl++) {
            const count = activations[lvl].length;
            for (let i = 0; i < count; i++) {
                const x = nodeX(lvl), y = nodeY(lvl, i, count);
                ctx.beginPath();
                ctx.fillStyle = getRGBA(Math.max(-1, Math.min(1, activations[lvl][i])));
                ctx.strokeStyle = '#1a1f2b';
                ctx.arc(x, y, 7, 0, Math.PI * 2);
                ctx.fill(); ctx.stroke();
                if (lvl === levelCount - 1) {
                    ctx.fillStyle = '#e5e9f0';
                    ctx.font = '10px sans-serif';
                    ctx.fillText(labelsOut[i] || '', x + 10, y + 3);
                }
            }
        }
    }

    // ---- car info panel ----
    function updateCarInfoPanel() {
        const c = state.selectedCar;
        if (!c || c.type !== 'AI') { carInfoPanel.innerHTML = '<p class="muted">Click a car to inspect it.</p>'; return; }
        carInfoPanel.innerHTML = `
            <h3>CAR #${c.id}</h3>
            <div class="stat-row"><span>Generation</span><b>${c.generation}</b></div>
            <div class="stat-row"><span>Fitness</span><b>${c.fitness.toFixed(0)}</b></div>
            <div class="stat-row"><span>Distance</span><b>${c.distanceTraveled.toFixed(0)} m</b></div>
            <div class="stat-row"><span>Speed</span><b>${(Math.abs(c.speed) * 22).toFixed(0)} km/h</b></div>
            <div class="stat-row"><span>Checkpoints</span><b>${c.currentCheckpointCount} / ${network.checkpoints.length}</b></div>
            <div class="stat-row"><span>Status</span><b>${c.damaged ? 'Crashed' : 'Driving'}</b></div>
        `;
    }

    // ---- main loop ----
    let lastTime = performance.now();
    let fpsSmoothed = 60;

    function frame(now) {
        requestAnimationFrame(frame);
        const rawDt = Math.min(0.05, (now - lastTime) / 1000);
        lastTime = now;
        fpsSmoothed = lerp(fpsSmoothed, 1 / Math.max(rawDt, 0.0001), 0.08);

        if (state.running) {
            // Multiple fixed physics steps per rendered frame is how "5x/10x/50x"
            // training speed works - the browser can still only paint once per
            // frame, but the simulation clock (and therefore learning) advances
            // faster. Capped so an enormous population + 50x doesn't hang the tab.
            const steps = Math.min(25, Math.round(state.speedMultiplier));
            const dt = 1 / 60;
            for (let s = 0; s < steps; s++) {
                for (const light of network.trafficLights) light.update(dt);
                for (const t of trafficCars) t.update(dt, network, evo.cars, network.buildings);
                evo.update(dt, trafficCars, network.buildings);
            }
        }
        camera.update();

        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = state.night ? '#0b1220' : '#bfe3ff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        camera.applyTransform(ctx);

        Renderer.drawWorld(ctx, network, state.night);
        if (state.showCheckpoints) Renderer.drawCheckpoints(ctx, network, state.selectedCar ? state.selectedCar.visitedCheckpoints : null);
        if (state.showBorders) Renderer.drawBorders(ctx, network);

        for (const t of trafficCars) t.draw(ctx, false, false, state.night);
        for (const c of evo.cars) {
            c.draw(ctx, state.showSensors && c === state.selectedCar, c === state.selectedCar, state.night);
        }

        // night overlay
        if (state.night) {
            ctx.fillStyle = 'rgba(5,10,25,0.35)';
            ctx.fillRect(-20000, -20000, 40000, 40000);
        }
        ctx.restore();

        // Minimap + NN visualiser + stats (screen space, always drawn)
        Renderer.drawMinimap(minimapCtx, minimapCanvas, network, evo.cars, trafficCars, state.selectedCar, camera);
        if (state.showNN) drawNetwork(nnCtx, nnCanvas, state.selectedCar);
        else nnCtx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

        const alive = evo.cars.filter(c => !c.damaged);
        const avgFit = evo.cars.reduce((s, c) => s + c.fitness, 0) / evo.cars.length;
        const best = evo.bestCar;
        statGen.textContent = evo.generation;
        statAlive.textContent = `${alive.length} / ${evo.cars.length}`;
        statBestDist.textContent = best ? best.distanceTraveled.toFixed(0) : '0';
        statBestFit.textContent = best ? best.fitness.toFixed(0) : '0';
        statAvgFit.textContent = isFinite(avgFit) ? avgFit.toFixed(0) : '0';
        statFps.textContent = fpsSmoothed.toFixed(0);

        if (state.selectedCar && state.selectedCar.damaged && camera.followCar === state.selectedCar) {
            camera.followCar = null;
        }
        updateCarInfoPanel();
    }
    requestAnimationFrame(frame);
})();
