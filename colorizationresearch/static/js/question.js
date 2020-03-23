function displayImg(){
    document.getElementById("display_img_btn").disabled = true;

    var img1 = document.getElementById("img1");
    var img2 = document.getElementById("img2");

    if(img1.style.visibility == "hidden" && img2.style.visibility == "hidden"){
        img1.style.visibility = "visible";
        img2.style.visibility = "visible";
    }
    setTimeout(function(){hiddenImg()},3000);
}

function hiddenImg(){
    var img1 = document.getElementById("img1");
    var img2 = document.getElementById("img2");

    if(img1.style.visibility == "visible" && img2.style.visibility == "visible"){
        img1.style.visibility = "hidden";
        img2.style.visibility = "hidden";
    }
}