const MATRIX_ELEMENTS = 25;
let patterns = 0;
let alphabet = [];
let weights = [];
let answerFound = false;
let energyPerIteration = [];
let testAlphabet = [
    [1,  1,  1,  1,  1, // Z
    -1, -1, -1,  1, -1,
    -1, -1,  1, -1, -1,
    -1,  1, -1, -1, -1,
     1,  1,  1,  1,  1],
    [1, -1, -1, -1,  1, // H
     1, -1, -1, -1,  1,
     1,  1,  1,  1,  1,
     1, -1, -1, -1,  1,
     1, -1, -1, -1,  1],
    [1, -1, -1, -1,  1, // X
    -1,  1, -1,  1, -1,
    -1, -1,  1, -1, -1,
    -1,  1, -1,  1, -1,
     1, -1, -1, -1,  1],
];
testPattern = []

function selectSquare(i, j) {
    let item = document.getElementsByClassName(`item-${i}-${j}`)[0];
    if(item.className.includes('on')) {
        item.className = item.className.replace('on', 'off');
    } else {
        item.className += ' on';
    }
}

function learnPattern(event) {
    event.preventDefault();
    event.stopPropagation();
    let items = Array.from(document.getElementsByClassName('item'));
    let matrix = [];
    items.forEach(item => {
        matrix.push(item.className.includes('on') ? 1 : -1);
    });
    patterns++;
    alphabet.push(matrix);
    document.getElementById('patterns').innerText = `Patrones aprendidos: ${patterns} de 4`
    if(patterns === 4) {
        document.getElementById('learn-button').setAttribute('disabled', 'true');
    }
    document.getElementById('start-button').disabled = false;
    document.getElementById('clear-button').disabled = false;
}

function generateRandomPattern() {
    let numberOfOnes = Math.floor(Math.random() * MATRIX_ELEMENTS);
    let randomIndexes = new Set();
    let out = [];
    for(let i = 0 ; i < 25 ; i++) {
        out.push(-1);
    }
    while(randomIndexes.size < numberOfOnes) {
        randomIndexes.add(Math.floor(Math.random() * MATRIX_ELEMENTS));
    }
    randomIndexes.forEach(idx => out[idx] = 1.0);
    return out;
}

async function startAlgorithm(type) {
    let pattern = [];
    if(type === 'test') {
        pattern = generateRandomPattern();
        alphabet = testAlphabet;
        drawTestPattern(pattern);
    } else {
        let items = Array.from(document.getElementsByClassName('item'));
        items.forEach(item => {
            pattern.push(item.className.includes('on') ? 1 : -1);
        });
    }
    initializeWeights();
    await hopfieldAlgorithm(pattern);
    if(type === 'test')
        alphabet = [];
}

function drawTestPattern(vector) {
    clearGrid(new Event('click'));
    let i = 0;
    for(let elem = 0 ; elem < vector.length ; elem++) {
        let j = elem % 5;
        let item = document.getElementsByClassName(`item-${i}-${j}`)[0];
        if(vector[elem] === 1)
            item.className += ' on';
        if(elem % 5 === 4)
            i++;
    }
}

function clearPatterns(event) {
    event.preventDefault();
    event.stopPropagation();
    alphabet = [];
    patterns = 0;
    document.getElementById('patterns').innerText = `Patrones aprendidos: ${patterns} de 4`;
    document.getElementById('start-button').disabled = true;
    document.getElementById('clear-button').disabled = true;
}

function sign(vector) {
    let out = [];
    for(let i = 0 ; i < vector.length ; i++) {
        if(vector[i] === 0.0)
            out[i] = 0.0;
        else if(vector[i] > 0.0)
            out[i] = 1.0;
        else
            out[i] = -1.0;
    }
    return out;
}

function matrixProduct(matrix, vector) {
    let ans = [];
    for(let row = 0 ; row < 25 ; row++) {
        let acum = 0.0;
        for(let col = 0 ; col < 25 ; col++) {
            acum += matrix[row][col] * vector[col];
        }
        ans.push(acum);
    }
    return ans;
}

function initializeWeights() {
    weights = [];
    for(let i = 0 ; i < 25 ; i++) {
        weights[i] = new Array(25);
        for(let wrow = 0 ; wrow < 25 ; wrow++) {
            weights[i][wrow] = 0.0;
        }
        for(let j = 0 ; j < 25 ; j++) {
            if(i !== j) {
                for(let k = 0 ; k < alphabet.length ; k++) {
                    weights[i][j] += alphabet[k][i] * alphabet[k][j];
                }
                weights[i][j] *= 1.0/alphabet[0].length;
            } else {
                weights[i][j] = 0.0;
            }
        }
    }
}

function canAdvance(S, prevS) {
    for(let letter = 0 ; letter < alphabet.length ; letter++) {
        if(JSON.stringify(alphabet[letter]) === JSON.stringify(S)) {
            answerFound = true;
            return false;
        }
    }
    if(JSON.stringify(S) === JSON.stringify(prevS))
        return false;
    return true;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getEnergyData() {
    return energyPerIteration;
}

function clearEnergies() {
    energyPerIteration = [];
}

function calculateEnergy(W, S) {
    let E = 0.0;
    for(let i = 0 ; i < W.length ; i++) {
        for(let j = 0 ; j < W.length ; j++) {
            E += W[i][j] * S[i] * S[j];
        }
    }
    E *= -0.5;
    if(energyPerIteration.length > 2) {
        let length = energyPerIteration.length;
        if(energyPerIteration[length - 1] === E && energyPerIteration[length - 2]) {
            return Infinity;
        }
    }
    energyPerIteration.push(E);
    console.log('calc', energyPerIteration);
    return E;
}

async function hopfieldAlgorithm(pattern) {
    if(pattern.length != alphabet[0].length) {
        console.log('Error');
        return;
    }
    clearEnergies();
    clearOutput();
    let S = pattern;
    let prevS = [];
    answerFound = false;
    let iterations = 0;
    console.log('Processing...');
    let startTime = new Date();
    fillInResult(S);
    let e = 0.0;
    document.getElementById('state').innerText = 'Procesando...';
    while(canAdvance(S, prevS) && iterations < 1000 && e !== Infinity) {
        prevS = S;
        e = calculateEnergy(weights, prevS);
        S = sign(matrixProduct(weights, prevS));
        fillInResult(S);
        iterations++;
        await sleep(1000);
        document.getElementById('iterations').innerText = `Número de iteraciones ${iterations}`;
    }
    calculateEnergy(weights, S);
    document.getElementById('state').innerText = 'Listo!';
    document.getElementById('final-result').innerHTML = answerFound ? 'Patrón reconocido' : 'Patrón no reconocido';
    let endTime = new Date();
    let diff = endTime - startTime - (1000 * iterations);
    document.getElementById('processing-time').innerHTML = 'Tiempo de ejecución: ' + diff + 'ms.';
    console.log('e final')
    if(e === Infinity)
        document.getElementById('limit-reached').innerHTML = 'El programa se ha cortado automáticamente ya que se ha llegado a un ciclo entre patrones con energía constante sin respuesta.'
    else if(iterations < 1000 && !answerFound)
        document.getElementById('limit-reached').innerHTML = 'El programa se ha cortado automáticamente ya que se ha llegado a un estado repetido sin respuesta.'
    else if(iterations === 1000 && !answerFound)
        document.getElementById('limit-reached').innerHTML = 'El programa se ha cortado automáticamente por alcanzar el límite de 1000 iteraciones sin reconocer un patrón.'
    fillInResult(S);
    console.log('final energies: ', getEnergyData());
}

function runTest(event) {
    event.preventDefault();
    event.stopPropagation();

}

function fillInResult(vector) {
    clearOutput();
    let i = 0;
    for(let elem = 0 ; elem < vector.length ; elem++) {
        let j = elem % 5;
        let item = document.getElementsByClassName(`itemO-${i}-${j}`)[0];
        if(vector[elem] === 1)
            item.className += ' on';
        if(elem % 5 === 4)
            i++;
    }
}

function clearGrid(event) {
    event.preventDefault();
    event.stopPropagation();
    let cells = document.getElementsByClassName('item');
    for(let i = 0 ; i < cells.length ; i++) {
        if(cells[i].className.includes('on'))
            cells[i].className = cells[i].className.replace('on', 'off');
    };
}

function clearOutput() {
    let cells = document.getElementsByClassName('itemO');
    for(let i = 0 ; i < cells.length ; i++) {
        if(cells[i].className.includes('on'))
            cells[i].className = cells[i].className.replace('on', 'off');
    };
}

function isEmpty() {
    return patterns === 0;
}