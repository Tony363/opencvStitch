var h3 = document.getElementsByTagName('h3')[0],
    start = document.getElementById('play'),
    clear = document.getElementById('clear'),
    seconds = 0, minutes = 0, hours = 0,
    t;

function add() {
    seconds++;
    if (seconds >= 60) {
        seconds = 0;
        minutes++;
        if (minutes >= 60) {
            minutes = 0;
            hours++;
        }
    }
    
    h3.textContent = (hours ? (hours > 9 ? hours : "0" + hours) : "00") + ":" + (minutes ? (minutes > 9 ? minutes : "0" + minutes) : "00") + ":" + (seconds > 9 ? seconds : "0" + seconds);

    timer();
}


function timer() {
    t = setTimeout(add, 1000);
}


/* Start button */
if (start !== null)
    start.onclick = timer;


/* Clear button */
if (clear !== null)
{
    clear.onclick = function() {
        h3.textContent = "00:00:00";
        seconds = 0; minutes = 0; hours = 0;
    }
    timer();
}
