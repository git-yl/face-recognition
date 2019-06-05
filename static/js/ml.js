var test_image = ''

function selectFile(){
    //触发 文件选择的click事件
    $("#file").trigger("click");
}
function getFileUrl(fileId) {
    var url;
    var file = document.getElementById(fileId);
    var agent = navigator.userAgent;
    if (agent.indexOf("MSIE")>=1) {
    url = file.value;
    } else if(agent.indexOf("Firefox")>0) {
    url = window.URL.createObjectURL(file.files.item(0));
    } else if(agent.indexOf("Chrome")>0) {
    url = window.URL.createObjectURL(file.files.item(0));
    }
    return url;
}

/* 获取文件的路径，将图片显示到界面*/
function getFilePath(){
    test_image = document.getElementById("file").files[0].name
    var imgPre = document.getElementById("test_image");
    imgPre.src = getFileUrl("file");
}

function test_one(type){
    if(test_image == ''){
        alert("请选择需要测试的图片")
    }else{
        data = type.split('_')
        if(data[0] == 'knn'){
            document.getElementById("algo_name").innerHTML=data[1];
        }
        getTestOne(type)
    }
}

function getTestOne(type) {
    $.ajax({
        type: "GET",
        contentType: "application/json",
        url: "/test-one",
        data: "test_image=" + test_image + "&type=" + type,
        timeout: 500000,
        success: function (data) {
            data = data.split("|")
            document.getElementById("test_image_label").innerHTML=data[1];
            if(data[0] == data[1]) {
                document.getElementById("test_image_result").innerHTML="预测成功";
            }else{
                document.getElementById("test_image_result").innerHTML="预测失败";
            }
            show_images(data[1])
        },
        error: function (XMLHttpRequest, textStatus, errorThrown) {
            alert("Get predict error!")
        }
    });
}
function show_images(data){
    document.getElementById('show_image1').src = "/image/" + data + "_" + "1"
    document.getElementById('show_image2').src = "/image/" + data + "_" + "2"
    document.getElementById('show_image3').src = "/image/" + data + "_" + "3"
    document.getElementById('show_image4').src = "/image/" + data + "_" + "4"
}

function getTestAll(type) {
    data = type.split('_')
    if(data[0] == 'knn'){
        document.getElementById("algo_name").innerHTML=data[1];
            document.getElementById("train_label_num_knn").innerHTML="";
            document.getElementById("test_label_num_knn").innerHTML="";
            document.getElementById("accuracy_knn").innerHTML="";
    }
    $.ajax({
        type: "GET",
        contentType: "application/json",
        url: "/test-all",
        data: "type=" + type,
        timeout: 500000,
        success: function (data) {
            data = data.split("|");
            type = type.split('_')[0]
            document.getElementById("train_label_num_"+type).innerHTML=data[0];
            document.getElementById("test_label_num_"+type).innerHTML=data[1];
            document.getElementById("accuracy_"+type).innerHTML=data[2];
        },
        error: function (XMLHttpRequest, textStatus, errorThrown) {
            alert("Get predict error!")
        }
    });
}