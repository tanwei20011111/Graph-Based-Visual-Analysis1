<template>
    <div>
        <div id="container" style="width: 1500px;height: 1500px;"> </div>
    </div>
</template>

<script>
import G6 from '@antv/g6';

export default {
    name: "Draw",
    data() {
        return {
            nodes: [
                { name: 0, color: 4 },
                { name: 1, color: 5 },
                { name: 2, color: 5 },
                { name: 3, color: 4 },
                { name: 4, color: 4 },
                { name: 5, color: 4 },
            ],
        };
    },
    mounted() {
        this.drawgraph()
    },
    methods: {
        drawgraph() {
            const data = {
                "nodes": [
                    {
                        "id": "0",
                        "name": "肺泡蛋白质沉积症",
                        "label": "Diseases"
                    },
                    {
                        "id": "157",
                        "name": "胸部CT检查",
                        "label": "Checks"
                    },
                    {
                        "id": "135",
                        "name": "肺活检",
                        "label": "Checks"
                    },
                    {
                        "id": "151",
                        "name": "支气管镜检查",
                        "label": "Checks"
                    },
                    {
                        "id": "1",
                        "name": "百日咳",
                        "label": "Diseases"
                    },
                    {
                        "id": "145",
                        "name": "耳、鼻、咽拭子细菌培养",
                        "label": "Checks"
                    },
                    {
                        "id": "147",
                        "name": "周围血白细胞计数及分类检验",
                        "label": "Checks"
                    },
                    {
                        "id": "148",
                        "name": "血常规",
                        "label": "Checks"
                    },
                    {
                        "id": "166",
                        "name": "酶联免疫吸附试验",
                        "label": "Checks"
                    },
                    {
                        "id": "140",
                        "name": "白细胞分类计数",
                        "label": "Checks"
                    },
                    {
                        "id": "2",
                        "name": "苯中毒",
                        "label": "Diseases"
                    },
                    {
                        "id": "142",
                        "name": "骨髓象分析",
                        "label": "Checks"
                    },
                    {
                        "id": "163",
                        "name": "先令氏指数",
                        "label": "Checks"
                    },
                    {
                        "id": "3",
                        "name": "喘息样支气管炎",
                        "label": "Diseases"
                    },
                    {
                        "id": "152",
                        "name": "肺部检查",
                        "label": "Checks"
                    },
                    {
                        "id": "164",
                        "name": "肺和胸膜听诊",
                        "label": "Checks"
                    },
                    {
                        "id": "153",
                        "name": "抗链球菌型M蛋白抗体",
                        "label": "Checks"
                    },
                    {
                        "id": "161",
                        "name": "抗链球菌壁多糖抗体",
                        "label": "Checks"
                    },
                    {
                        "id": "4",
                        "name": "成人呼吸窘迫综合征",
                        "label": "Diseases"
                    },
                    {
                        "id": "136",
                        "name": "呼吸肌功能测定",
                        "label": "Checks"
                    },
                    {
                        "id": "165",
                        "name": "血浆蛋白C抗原",
                        "label": "Checks"
                    },
                    {
                        "id": "162",
                        "name": "肺泡气-动脉血氧分压差",
                        "label": "Checks"
                    },
                    {
                        "id": "141",
                        "name": "肺毛细血管楔压",
                        "label": "Checks"
                    },
                    {
                        "id": "5",
                        "name": "大量羊水吸入",
                        "label": "Diseases"
                    },
                    {
                        "id": "159",
                        "name": "胸部透视",
                        "label": "Checks"
                    },
                    {
                        "id": "149",
                        "name": "胸部平片",
                        "label": "Checks"
                    },
                    {
                        "id": "6",
                        "name": "单纯性肺嗜酸粒细胞浸润症",
                        "label": "Diseases"
                    },
                    {
                        "id": "156",
                        "name": "痰液中细胞分类",
                        "label": "Checks"
                    },
                    {
                        "id": "132",
                        "name": "痰液中寄生虫和虫卵",
                        "label": "Checks"
                    },
                    {
                        "id": "7",
                        "name": "大叶性肺炎",
                        "label": "Diseases"
                    },
                    {
                        "id": "154",
                        "name": "Optochin敏感试验",
                        "label": "Checks"
                    },
                    {
                        "id": "155",
                        "name": "小白鼠毒力试验",
                        "label": "Checks"
                    },
                    {
                        "id": "133",
                        "name": "痰培养",
                        "label": "Checks"
                    },
                    {
                        "id": "131",
                        "name": "肺活量体重指数",
                        "label": "Checks"
                    },
                    {
                        "id": "137",
                        "name": "免疫电泳",
                        "label": "Checks"
                    },
                    {
                        "id": "160",
                        "name": "痰液细菌涂片检查",
                        "label": "Checks"
                    },
                    {
                        "id": "8",
                        "name": "大楼病综合征",
                        "label": "Diseases"
                    },
                    {
                        "id": "158",
                        "name": "钼靶X线检查",
                        "label": "Checks"
                    },
                    {
                        "id": "143",
                        "name": "CT检查",
                        "label": "Checks"
                    },
                    {
                        "id": "9",
                        "name": "二硫化碳中毒",
                        "label": "Diseases"
                    },
                    {
                        "id": "167",
                        "name": "尿常规",
                        "label": "Checks"
                    },
                    {
                        "id": "138",
                        "name": "肾功能检查",
                        "label": "Checks"
                    },
                    {
                        "id": "139",
                        "name": "神经系统检查",
                        "label": "Checks"
                    },
                    {
                        "id": "144",
                        "name": "眼底荧光血管造影",
                        "label": "Checks"
                    },
                    {
                        "id": "146",
                        "name": "肌电图",
                        "label": "Checks"
                    },
                    {
                        "id": "150",
                        "name": "脑电图检查",
                        "label": "Checks"
                    },
                    {
                        "id": "134",
                        "name": "心肺功能运动试验（CPET）",
                        "label": "Checks"
                    },
                    {
                        "id": "108",
                        "name": "清蒸鸡蛋羹",
                        "label": "Foods"
                    },
                    {
                        "id": "80",
                        "name": "百合双耳鸡蛋羹",
                        "label": "Foods"
                    },
                    {
                        "id": "54",
                        "name": "排骨汤",
                        "label": "Foods"
                    },
                    {
                        "id": "114",
                        "name": "罗汉果雪耳鸡汤",
                        "label": "Foods"
                    },
                    {
                        "id": "120",
                        "name": "小黄瓜凉拌面",
                        "label": "Foods"
                    },
                    {
                        "id": "95",
                        "name": "黄瓜三丝汤",
                        "label": "Foods"
                    },
                    {
                        "id": "100",
                        "name": "黄瓜拌兔丝",
                        "label": "Foods"
                    },
                    {
                        "id": "119",
                        "name": "黄瓜拌皮丝",
                        "label": "Foods"
                    },
                    {
                        "id": "90",
                        "name": "豆腐干炒韭菜",
                        "label": "Foods"
                    },
                    {
                        "id": "103",
                        "name": "素炒小白菜",
                        "label": "Foods"
                    },
                    {
                        "id": "65",
                        "name": "白菜蛋花粥",
                        "label": "Foods"
                    },
                    {
                        "id": "104",
                        "name": "紫苏粥",
                        "label": "Foods"
                    },
                    {
                        "id": "82",
                        "name": "菊花芦根茶",
                        "label": "Foods"
                    },
                    {
                        "id": "130",
                        "name": "生芦根粥",
                        "label": "Foods"
                    },
                    {
                        "id": "61",
                        "name": "鲜芦根粥",
                        "label": "Foods"
                    },
                    {
                        "id": "87",
                        "name": "薏仁党参粥",
                        "label": "Foods"
                    },
                    {
                        "id": "62",
                        "name": "党参粥",
                        "label": "Foods"
                    },
                    {
                        "id": "67",
                        "name": "黄芪粥",
                        "label": "Foods"
                    },
                    {
                        "id": "68",
                        "name": "鸡蛋木耳粥",
                        "label": "Foods"
                    },
                    {
                        "id": "66",
                        "name": "百合糖粥",
                        "label": "Foods"
                    },
                    {
                        "id": "115",
                        "name": "百合粥",
                        "label": "Foods"
                    },
                    {
                        "id": "69",
                        "name": "百合花生汤",
                        "label": "Foods"
                    },
                    {
                        "id": "110",
                        "name": "首乌百合粥",
                        "label": "Foods"
                    },
                    {
                        "id": "112",
                        "name": "薏米莲子粥",
                        "label": "Foods"
                    },
                    {
                        "id": "111",
                        "name": "牛奶玉米汤",
                        "label": "Foods"
                    },
                    {
                        "id": "86",
                        "name": "萝卜豆腐汤",
                        "label": "Foods"
                    },
                    {
                        "id": "78",
                        "name": "瓜粒杂锦汤",
                        "label": "Foods"
                    },
                    {
                        "id": "92",
                        "name": "奶汤锅子鱼",
                        "label": "Foods"
                    },
                    {
                        "id": "122",
                        "name": "酱豆腐汁烧猪肉",
                        "label": "Foods"
                    },
                    {
                        "id": "116",
                        "name": "百合汤",
                        "label": "Foods"
                    },
                    {
                        "id": "109",
                        "name": "山楂百合汤",
                        "label": "Foods"
                    },
                    {
                        "id": "84",
                        "name": "番茄鸡蛋煎饼",
                        "label": "Foods"
                    },
                    {
                        "id": "72",
                        "name": "凉拌番茄",
                        "label": "Foods"
                    },
                    {
                        "id": "56",
                        "name": "番茄猪肝瘦肉汤",
                        "label": "Foods"
                    },
                    {
                        "id": "55",
                        "name": "番茄牛肉",
                        "label": "Foods"
                    },
                    {
                        "id": "75",
                        "name": "生姜乌梅饮",
                        "label": "Foods"
                    },
                    {
                        "id": "79",
                        "name": "乌梅粥",
                        "label": "Foods"
                    },
                    {
                        "id": "123",
                        "name": "山楂乌梅饮",
                        "label": "Foods"
                    },
                    {
                        "id": "117",
                        "name": "山楂红糖粥",
                        "label": "Foods"
                    },
                    {
                        "id": "77",
                        "name": "山楂桑椹粥",
                        "label": "Foods"
                    },
                    {
                        "id": "93",
                        "name": "小麦粥",
                        "label": "Foods"
                    },
                    {
                        "id": "57",
                        "name": "荞麦粥",
                        "label": "Foods"
                    },
                    {
                        "id": "58",
                        "name": "杂粮粥",
                        "label": "Foods"
                    },
                    {
                        "id": "107",
                        "name": "蔬菜杂烩粥",
                        "label": "Foods"
                    },
                    {
                        "id": "102",
                        "name": "花生杏仁粥",
                        "label": "Foods"
                    },
                    {
                        "id": "97",
                        "name": "豆薯小米粥",
                        "label": "Foods"
                    },
                    {
                        "id": "71",
                        "name": "羊腩萝卜丝粥",
                        "label": "Foods"
                    },
                    {
                        "id": "128",
                        "name": "绿豆南瓜汤",
                        "label": "Foods"
                    },
                    {
                        "id": "98",
                        "name": "螃蟹",
                        "label": "Foods"
                    },
                    {
                        "id": "64",
                        "name": "海蟹",
                        "label": "Foods"
                    },
                    {
                        "id": "88",
                        "name": "海虾",
                        "label": "Foods"
                    },
                    {
                        "id": "89",
                        "name": "海螺",
                        "label": "Foods"
                    },
                    {
                        "id": "73",
                        "name": "海参（水浸）",
                        "label": "Foods"
                    },
                    {
                        "id": "129",
                        "name": "辣椒(青、尖)",
                        "label": "Foods"
                    },
                    {
                        "id": "63",
                        "name": "猪肉(肥瘦)",
                        "label": "Foods"
                    },
                    {
                        "id": "105",
                        "name": "啤酒",
                        "label": "Foods"
                    },
                    {
                        "id": "127",
                        "name": "白酒",
                        "label": "Foods"
                    },
                    {
                        "id": "96",
                        "name": "洋葱",
                        "label": "Foods"
                    },
                    {
                        "id": "60",
                        "name": "辣椒(红、尖、干)",
                        "label": "Foods"
                    },
                    {
                        "id": "70",
                        "name": "韭菜",
                        "label": "Foods"
                    },
                    {
                        "id": "81",
                        "name": "南瓜子仁",
                        "label": "Foods"
                    },
                    {
                        "id": "83",
                        "name": "圆白菜",
                        "label": "Foods"
                    },
                    {
                        "id": "125",
                        "name": "樱桃番茄",
                        "label": "Foods"
                    },
                    {
                        "id": "121",
                        "name": "小白菜",
                        "label": "Foods"
                    },
                    {
                        "id": "113",
                        "name": "鸡蛋",
                        "label": "Foods"
                    },
                    {
                        "id": "53",
                        "name": "大豆",
                        "label": "Foods"
                    },
                    {
                        "id": "59",
                        "name": "猪肉(瘦)",
                        "label": "Foods"
                    },
                    {
                        "id": "126",
                        "name": "海蚌",
                        "label": "Foods"
                    },
                    {
                        "id": "99",
                        "name": "赤贝",
                        "label": "Foods"
                    },
                    {
                        "id": "94",
                        "name": "莲子",
                        "label": "Foods"
                    },
                    {
                        "id": "74",
                        "name": "生菜",
                        "label": "Foods"
                    },
                    {
                        "id": "76",
                        "name": "山药",
                        "label": "Foods"
                    },
                    {
                        "id": "101",
                        "name": "空心菜",
                        "label": "Foods"
                    },
                    {
                        "id": "106",
                        "name": "栗子（熟）",
                        "label": "Foods"
                    },
                    {
                        "id": "91",
                        "name": "鲫鱼",
                        "label": "Foods"
                    },
                    {
                        "id": "118",
                        "name": "油菜",
                        "label": "Foods"
                    },
                    {
                        "id": "85",
                        "name": "南瓜",
                        "label": "Foods"
                    },
                    {
                        "id": "124",
                        "name": "苦瓜",
                        "label": "Foods"
                    },
                    {
                        "id": "173",
                        "name": "呼吸内科",
                        "label": "Departments"
                    },
                    {
                        "id": "171",
                        "name": "内科",
                        "label": "Departments"
                    },
                    {
                        "id": "172",
                        "name": "小儿内科",
                        "label": "Departments"
                    },
                    {
                        "id": "169",
                        "name": "儿科",
                        "label": "Departments"
                    },
                    {
                        "id": "168",
                        "name": "其他综合",
                        "label": "Departments"
                    },
                    {
                        "id": "174",
                        "name": "其他科室",
                        "label": "Departments"
                    },
                    {
                        "id": "41",
                        "name": "穿心莲内酯片",
                        "label": "Drugs"
                    },
                    {
                        "id": "37",
                        "name": "百咳静糖浆",
                        "label": "Drugs"
                    },
                    {
                        "id": "51",
                        "name": "布美他尼片",
                        "label": "Drugs"
                    },
                    {
                        "id": "22",
                        "name": "十一味金色丸",
                        "label": "Drugs"
                    },
                    {
                        "id": "27",
                        "name": "布地奈德气雾剂",
                        "label": "Drugs"
                    },
                    {
                        "id": "39",
                        "name": "喷托维林氯化铵糖浆",
                        "label": "Drugs"
                    },
                    {
                        "id": "47",
                        "name": "人血白蛋白",
                        "label": "Drugs"
                    },
                    {
                        "id": "50",
                        "name": "注射用还原型谷胱甘肽钠",
                        "label": "Drugs"
                    },
                    {
                        "id": "42",
                        "name": "头孢泊肟酯胶囊",
                        "label": "Drugs"
                    },
                    {
                        "id": "19",
                        "name": "头孢氨苄甲氧苄啶片",
                        "label": "Drugs"
                    },
                    {
                        "id": "21",
                        "name": "乳酸左氧氟沙星片",
                        "label": "Drugs"
                    },
                    {
                        "id": "49",
                        "name": "阿奇霉素片",
                        "label": "Drugs"
                    },
                    {
                        "id": "31",
                        "name": "云南花粉片",
                        "label": "Drugs"
                    },
                    {
                        "id": "44",
                        "name": "注射用盐酸溴己新",
                        "label": "Drugs"
                    },
                    {
                        "id": "371",
                        "name": "紫绀",
                        "label": "Symptoms"
                    },
                    {
                        "id": "356",
                        "name": "胸痛",
                        "label": "Symptoms"
                    },
                    {
                        "id": "376",
                        "name": "呼吸困难",
                        "label": "Symptoms"
                    },
                    {
                        "id": "346",
                        "name": "乏力",
                        "label": "Symptoms"
                    },
                    {
                        "id": "367",
                        "name": "毓卓",
                        "label": "Symptoms"
                    },
                    {
                        "id": "366",
                        "name": "吸气时有蝉鸣音",
                        "label": "Symptoms"
                    },
                    {
                        "id": "329",
                        "name": "痉挛性咳嗽",
                        "label": "Symptoms"
                    },
                    {
                        "id": "325",
                        "name": "胸闷",
                        "label": "Symptoms"
                    },
                    {
                        "id": "358",
                        "name": "肺阴虚",
                        "label": "Symptoms"
                    },
                    {
                        "id": "348",
                        "name": "抽搐",
                        "label": "Symptoms"
                    },
                    {
                        "id": "353",
                        "name": "低热",
                        "label": "Symptoms"
                    },
                    {
                        "id": "335",
                        "name": "闫鹏辉",
                        "label": "Symptoms"
                    },
                    {
                        "id": "339",
                        "name": "惊厥",
                        "label": "Symptoms"
                    },
                    {
                        "id": "336",
                        "name": "恶心",
                        "label": "Symptoms"
                    },
                    {
                        "id": "357",
                        "name": "感觉障碍",
                        "label": "Symptoms"
                    },
                    {
                        "id": "370",
                        "name": "耸肩喘息",
                        "label": "Symptoms"
                    },
                    {
                        "id": "354",
                        "name": "哮鸣音",
                        "label": "Symptoms"
                    },
                    {
                        "id": "337",
                        "name": "纤毛上皮细胞损伤脱落",
                        "label": "Symptoms"
                    },
                    {
                        "id": "375",
                        "name": "变应性咳嗽",
                        "label": "Symptoms"
                    },
                    {
                        "id": "365",
                        "name": "化学性支气管炎",
                        "label": "Symptoms"
                    },
                    {
                        "id": "332",
                        "name": "喘息",
                        "label": "Symptoms"
                    },
                    {
                        "id": "361",
                        "name": "冬春季的慢性咳...",
                        "label": "Symptoms"
                    },
                    {
                        "id": "331",
                        "name": "咳嗽伴哮鸣音",
                        "label": "Symptoms"
                    },
                    {
                        "id": "360",
                        "name": "心源性呼吸窘迫",
                        "label": "Symptoms"
                    },
                    {
                        "id": "372",
                        "name": "面色青紫",
                        "label": "Symptoms"
                    },
                    {
                        "id": "374",
                        "name": "口唇青紫",
                        "label": "Symptoms"
                    },
                    {
                        "id": "373",
                        "name": "肺纹理增粗",
                        "label": "Symptoms"
                    },
                    {
                        "id": "333",
                        "name": "出生后即有持续青紫",
                        "label": "Symptoms"
                    },
                    {
                        "id": "359",
                        "name": "口唇和甲床略带青紫",
                        "label": "Symptoms"
                    },
                    {
                        "id": "368",
                        "name": "气急",
                        "label": "Symptoms"
                    },
                    {
                        "id": "343",
                        "name": "嗜酸性粒细胞增多",
                        "label": "Symptoms"
                    },
                    {
                        "id": "338",
                        "name": "咽部异物感",
                        "label": "Symptoms"
                    },
                    {
                        "id": "362",
                        "name": "胸闷憋气",
                        "label": "Symptoms"
                    },
                    {
                        "id": "377",
                        "name": "咯血伴发热",
                        "label": "Symptoms"
                    },
                    {
                        "id": "364",
                        "name": "湿啰音",
                        "label": "Symptoms"
                    },
                    {
                        "id": "328",
                        "name": "发烧",
                        "label": "Symptoms"
                    },
                    {
                        "id": "344",
                        "name": "咳铁锈色痰",
                        "label": "Symptoms"
                    },
                    {
                        "id": "345",
                        "name": "急性面容",
                        "label": "Symptoms"
                    },
                    {
                        "id": "355",
                        "name": "呼吸音减弱",
                        "label": "Symptoms"
                    },
                    {
                        "id": "330",
                        "name": "紧张性头晕",
                        "label": "Symptoms"
                    },
                    {
                        "id": "341",
                        "name": "紧张性头痛",
                        "label": "Symptoms"
                    },
                    {
                        "id": "352",
                        "name": "间歇性头晕",
                        "label": "Symptoms"
                    },
                    {
                        "id": "334",
                        "name": "鼻塞",
                        "label": "Symptoms"
                    },
                    {
                        "id": "326",
                        "name": "头晕",
                        "label": "Symptoms"
                    },
                    {
                        "id": "347",
                        "name": "眼睛痒",
                        "label": "Symptoms"
                    },
                    {
                        "id": "350",
                        "name": "目赤",
                        "label": "Symptoms"
                    },
                    {
                        "id": "369",
                        "name": "间歇性头痛",
                        "label": "Symptoms"
                    },
                    {
                        "id": "363",
                        "name": "昏迷",
                        "label": "Symptoms"
                    },
                    {
                        "id": "327",
                        "name": "腱反射消失",
                        "label": "Symptoms"
                    },
                    {
                        "id": "351",
                        "name": "呕吐",
                        "label": "Symptoms"
                    },
                    {
                        "id": "340",
                        "name": "谵妄",
                        "label": "Symptoms"
                    },
                    {
                        "id": "342",
                        "name": "多发性神经炎",
                        "label": "Symptoms"
                    },
                    {
                        "id": "349",
                        "name": "浅感觉减退或缺失",
                        "label": "Symptoms"
                    },
                    {
                        "id": "378",
                        "name": "多重肺部感染",
                        "label": "Diseases"
                    },
                    {
                        "id": "379",
                        "name": "肺不张",
                        "label": "Diseases"
                    },
                    {
                        "id": "380",
                        "name": "贫血",
                        "label": "Diseases"
                    },
                    {
                        "id": "381",
                        "name": "支气管哮喘",
                        "label": "Diseases"
                    },
                    {
                        "id": "382",
                        "name": "细菌性肺炎",
                        "label": "Diseases"
                    },
                    {
                        "id": "383",
                        "name": "呼吸衰竭",
                        "label": "Diseases"
                    },
                    {
                        "id": "384",
                        "name": "胆道蛔虫病",
                        "label": "Diseases"
                    },
                    {
                        "id": "385",
                        "name": "脓胸",
                        "label": "Diseases"
                    },
                    {
                        "id": "386",
                        "name": "抑郁症",
                        "label": "Diseases"
                    },
                    {
                        "id": "170",
                        "name": "急诊科",
                        "label": "Departments"
                    },
                    {
                        "id": "264",
                        "name": "惠普森穿心莲内酯片",
                        "label": "Producers"
                    },
                    {
                        "id": "195",
                        "name": "北京同仁堂百咳静糖浆",
                        "label": "Producers"
                    },
                    {
                        "id": "286",
                        "name": "邦琪药业百咳静糖浆",
                        "label": "Producers"
                    },
                    {
                        "id": "308",
                        "name": "东新药业百咳静糖浆",
                        "label": "Producers"
                    },
                    {
                        "id": "261",
                        "name": "永定制药百咳静糖浆",
                        "label": "Producers"
                    },
                    {
                        "id": "231",
                        "name": "佐今明百咳静糖浆",
                        "label": "Producers"
                    },
                    {
                        "id": "213",
                        "name": "桂林南药布美他尼片",
                        "label": "Producers"
                    },
                    {
                        "id": "273",
                        "name": "雄巴拉曲神水十一味金色丸",
                        "label": "Producers"
                    },
                    {
                        "id": "235",
                        "name": "西藏甘露十一味金色丸",
                        "label": "Producers"
                    },
                    {
                        "id": "258",
                        "name": "信谊布地奈德气雾剂",
                        "label": "Producers"
                    },
                    {
                        "id": "237",
                        "name": "桑海制药喷托维林氯化铵糖浆",
                        "label": "Producers"
                    },
                    {
                        "id": "203",
                        "name": "赣南制药喷托维林氯化铵糖浆",
                        "label": "Producers"
                    },
                    {
                        "id": "183",
                        "name": "莱士人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "198",
                        "name": "卫光生物人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "208",
                        "name": "泰邦生物人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "302",
                        "name": "绿十字",
                        "label": "Producers"
                    },
                    {
                        "id": "305",
                        "name": "松泰斯",
                        "label": "Producers"
                    },
                    {
                        "id": "236",
                        "name": "武汉生物人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "225",
                        "name": "德源生物人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "259",
                        "name": "山西康宝人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "301",
                        "name": "蜀阳药业人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "216",
                        "name": "双林生物人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "298",
                        "name": "蓉生药业人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "217",
                        "name": "广东卫伦人血白蛋白",
                        "label": "Producers"
                    },
                    {
                        "id": "223",
                        "name": "亮博",
                        "label": "Producers"
                    },
                    {
                        "id": "275",
                        "name": "白云山光华头孢氨苄甲氧苄啶",
                        "label": "Producers"
                    },
                    {
                        "id": "209",
                        "name": "华瑞联合制药头孢氨苄甲氧苄",
                        "label": "Producers"
                    },
                    {
                        "id": "224",
                        "name": "明水药业头孢氨苄甲氧苄啶片",
                        "label": "Producers"
                    },
                    {
                        "id": "306",
                        "name": "圣大（张家口）头孢氨苄甲氧",
                        "label": "Producers"
                    },
                    {
                        "id": "196",
                        "name": "哈药总厂头孢氨苄甲氧苄啶片",
                        "label": "Producers"
                    },
                    {
                        "id": "262",
                        "name": "云门药业头孢氨苄甲氧苄啶片",
                        "label": "Producers"
                    },
                    {
                        "id": "211",
                        "name": "搏沃欣",
                        "label": "Producers"
                    },
                    {
                        "id": "278",
                        "name": "宜昌长江乳酸左氧氟沙星片",
                        "label": "Producers"
                    },
                    {
                        "id": "282",
                        "name": "希舒美",
                        "label": "Producers"
                    },
                    {
                        "id": "269",
                        "name": "维宏",
                        "label": "Producers"
                    },
                    {
                        "id": "240",
                        "name": "辅仁药业阿奇霉素片",
                        "label": "Producers"
                    },
                    {
                        "id": "322",
                        "name": "汇仁药业阿奇霉素片",
                        "label": "Producers"
                    },
                    {
                        "id": "242",
                        "name": "爱普森药业阿奇霉素片",
                        "label": "Producers"
                    },
                    {
                        "id": "238",
                        "name": "海王阿奇霉素片",
                        "label": "Producers"
                    },
                    {
                        "id": "271",
                        "name": "云南白药云南花粉片",
                        "label": "Producers"
                    },
                    {
                        "id": "229",
                        "name": "丰原制药注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "287",
                        "name": "普德药业注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "274",
                        "name": "海斯制药注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "200",
                        "name": "国药国瑞注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "189",
                        "name": "傲群",
                        "label": "Producers"
                    },
                    {
                        "id": "315",
                        "name": "智同生物注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "316",
                        "name": "澳利达奈德注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "324",
                        "name": "一品红注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "210",
                        "name": "海鲸药业注射用盐酸溴己新",
                        "label": "Producers"
                    },
                    {
                        "id": "176",
                        "name": "益佳清",
                        "label": "Producers"
                    },
                    {
                        "id": "260",
                        "name": "伏枝",
                        "label": "Producers"
                    }
                ],
                "edges": [
                    {
                        "source": "0",
                        "target": "157",
                        "relation": "need_check"
                    },
                    {
                        "source": "0",
                        "target": "135",
                        "relation": "need_check"
                    },
                    {
                        "source": "0",
                        "target": "151",
                        "relation": "need_check"
                    },
                    {
                        "source": "0",
                        "target": "371",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "0",
                        "target": "356",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "0",
                        "target": "376",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "0",
                        "target": "346",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "0",
                        "target": "367",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "0",
                        "target": "378",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "0",
                        "target": "173",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "4",
                        "target": "157",
                        "relation": "need_check"
                    },
                    {
                        "source": "5",
                        "target": "157",
                        "relation": "need_check"
                    },
                    {
                        "source": "1",
                        "target": "145",
                        "relation": "need_check"
                    },
                    {
                        "source": "1",
                        "target": "147",
                        "relation": "need_check"
                    },
                    {
                        "source": "1",
                        "target": "148",
                        "relation": "need_check"
                    },
                    {
                        "source": "1",
                        "target": "166",
                        "relation": "need_check"
                    },
                    {
                        "source": "1",
                        "target": "140",
                        "relation": "need_check"
                    },
                    {
                        "source": "1",
                        "target": "108",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "80",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "54",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "114",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "120",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "95",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "100",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "119",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "1",
                        "target": "98",
                        "relation": "no_eat"
                    },
                    {
                        "source": "1",
                        "target": "64",
                        "relation": "no_eat"
                    },
                    {
                        "source": "1",
                        "target": "88",
                        "relation": "no_eat"
                    },
                    {
                        "source": "1",
                        "target": "89",
                        "relation": "no_eat"
                    },
                    {
                        "source": "1",
                        "target": "81",
                        "relation": "do_eat"
                    },
                    {
                        "source": "1",
                        "target": "83",
                        "relation": "do_eat"
                    },
                    {
                        "source": "1",
                        "target": "125",
                        "relation": "do_eat"
                    },
                    {
                        "source": "1",
                        "target": "121",
                        "relation": "do_eat"
                    },
                    {
                        "source": "1",
                        "target": "41",
                        "relation": "common_drug"
                    },
                    {
                        "source": "1",
                        "target": "37",
                        "relation": "common_drug"
                    },
                    {
                        "source": "1",
                        "target": "366",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "329",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "325",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "358",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "348",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "353",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "335",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "339",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "1",
                        "target": "379",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "1",
                        "target": "172",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "2",
                        "target": "148",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "148",
                        "relation": "need_check"
                    },
                    {
                        "source": "8",
                        "target": "148",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "148",
                        "relation": "need_check"
                    },
                    {
                        "source": "3",
                        "target": "166",
                        "relation": "need_check"
                    },
                    {
                        "source": "2",
                        "target": "142",
                        "relation": "need_check"
                    },
                    {
                        "source": "2",
                        "target": "163",
                        "relation": "need_check"
                    },
                    {
                        "source": "2",
                        "target": "90",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "2",
                        "target": "103",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "2",
                        "target": "65",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "2",
                        "target": "64",
                        "relation": "no_eat"
                    },
                    {
                        "source": "2",
                        "target": "88",
                        "relation": "no_eat"
                    },
                    {
                        "source": "2",
                        "target": "73",
                        "relation": "no_eat"
                    },
                    {
                        "source": "2",
                        "target": "129",
                        "relation": "no_eat"
                    },
                    {
                        "source": "2",
                        "target": "113",
                        "relation": "do_eat"
                    },
                    {
                        "source": "2",
                        "target": "53",
                        "relation": "do_eat"
                    },
                    {
                        "source": "2",
                        "target": "59",
                        "relation": "do_eat"
                    },
                    {
                        "source": "2",
                        "target": "125",
                        "relation": "do_eat"
                    },
                    {
                        "source": "2",
                        "target": "51",
                        "relation": "common_drug"
                    },
                    {
                        "source": "2",
                        "target": "22",
                        "relation": "common_drug"
                    },
                    {
                        "source": "2",
                        "target": "336",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "2",
                        "target": "348",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "2",
                        "target": "357",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "2",
                        "target": "380",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "2",
                        "target": "170",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "3",
                        "target": "152",
                        "relation": "need_check"
                    },
                    {
                        "source": "3",
                        "target": "164",
                        "relation": "need_check"
                    },
                    {
                        "source": "3",
                        "target": "153",
                        "relation": "need_check"
                    },
                    {
                        "source": "3",
                        "target": "161",
                        "relation": "need_check"
                    },
                    {
                        "source": "3",
                        "target": "104",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "82",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "130",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "61",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "87",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "62",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "67",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "68",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "3",
                        "target": "64",
                        "relation": "no_eat"
                    },
                    {
                        "source": "3",
                        "target": "88",
                        "relation": "no_eat"
                    },
                    {
                        "source": "3",
                        "target": "89",
                        "relation": "no_eat"
                    },
                    {
                        "source": "3",
                        "target": "63",
                        "relation": "no_eat"
                    },
                    {
                        "source": "3",
                        "target": "113",
                        "relation": "do_eat"
                    },
                    {
                        "source": "3",
                        "target": "126",
                        "relation": "do_eat"
                    },
                    {
                        "source": "3",
                        "target": "99",
                        "relation": "do_eat"
                    },
                    {
                        "source": "3",
                        "target": "83",
                        "relation": "do_eat"
                    },
                    {
                        "source": "3",
                        "target": "27",
                        "relation": "common_drug"
                    },
                    {
                        "source": "3",
                        "target": "39",
                        "relation": "common_drug"
                    },
                    {
                        "source": "3",
                        "target": "370",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "354",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "337",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "375",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "365",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "332",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "361",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "331",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "3",
                        "target": "381",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "3",
                        "target": "173",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "5",
                        "target": "152",
                        "relation": "need_check"
                    },
                    {
                        "source": "4",
                        "target": "136",
                        "relation": "need_check"
                    },
                    {
                        "source": "4",
                        "target": "165",
                        "relation": "need_check"
                    },
                    {
                        "source": "4",
                        "target": "162",
                        "relation": "need_check"
                    },
                    {
                        "source": "4",
                        "target": "141",
                        "relation": "need_check"
                    },
                    {
                        "source": "4",
                        "target": "66",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "115",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "69",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "110",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "112",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "111",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "86",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "78",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "4",
                        "target": "105",
                        "relation": "no_eat"
                    },
                    {
                        "source": "4",
                        "target": "127",
                        "relation": "no_eat"
                    },
                    {
                        "source": "4",
                        "target": "96",
                        "relation": "no_eat"
                    },
                    {
                        "source": "4",
                        "target": "60",
                        "relation": "no_eat"
                    },
                    {
                        "source": "4",
                        "target": "94",
                        "relation": "do_eat"
                    },
                    {
                        "source": "4",
                        "target": "74",
                        "relation": "do_eat"
                    },
                    {
                        "source": "4",
                        "target": "76",
                        "relation": "do_eat"
                    },
                    {
                        "source": "4",
                        "target": "101",
                        "relation": "do_eat"
                    },
                    {
                        "source": "4",
                        "target": "47",
                        "relation": "common_drug"
                    },
                    {
                        "source": "4",
                        "target": "50",
                        "relation": "common_drug"
                    },
                    {
                        "source": "4",
                        "target": "376",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "4",
                        "target": "371",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "4",
                        "target": "360",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "4",
                        "target": "382",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "4",
                        "target": "173",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "5",
                        "target": "159",
                        "relation": "need_check"
                    },
                    {
                        "source": "5",
                        "target": "149",
                        "relation": "need_check"
                    },
                    {
                        "source": "5",
                        "target": "372",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "5",
                        "target": "376",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "5",
                        "target": "374",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "5",
                        "target": "373",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "5",
                        "target": "333",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "5",
                        "target": "359",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "5",
                        "target": "368",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "5",
                        "target": "383",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "5",
                        "target": "172",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "6",
                        "target": "149",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "149",
                        "relation": "need_check"
                    },
                    {
                        "source": "6",
                        "target": "156",
                        "relation": "need_check"
                    },
                    {
                        "source": "6",
                        "target": "132",
                        "relation": "need_check"
                    },
                    {
                        "source": "6",
                        "target": "42",
                        "relation": "common_drug"
                    },
                    {
                        "source": "6",
                        "target": "19",
                        "relation": "common_drug"
                    },
                    {
                        "source": "6",
                        "target": "343",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "6",
                        "target": "338",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "6",
                        "target": "362",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "6",
                        "target": "377",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "6",
                        "target": "325",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "6",
                        "target": "346",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "6",
                        "target": "384",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "6",
                        "target": "173",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "7",
                        "target": "154",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "155",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "133",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "131",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "137",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "160",
                        "relation": "need_check"
                    },
                    {
                        "source": "7",
                        "target": "92",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "122",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "116",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "109",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "84",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "72",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "56",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "55",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "7",
                        "target": "96",
                        "relation": "no_eat"
                    },
                    {
                        "source": "7",
                        "target": "129",
                        "relation": "no_eat"
                    },
                    {
                        "source": "7",
                        "target": "60",
                        "relation": "no_eat"
                    },
                    {
                        "source": "7",
                        "target": "70",
                        "relation": "no_eat"
                    },
                    {
                        "source": "7",
                        "target": "106",
                        "relation": "do_eat"
                    },
                    {
                        "source": "7",
                        "target": "91",
                        "relation": "do_eat"
                    },
                    {
                        "source": "7",
                        "target": "59",
                        "relation": "do_eat"
                    },
                    {
                        "source": "7",
                        "target": "118",
                        "relation": "do_eat"
                    },
                    {
                        "source": "7",
                        "target": "21",
                        "relation": "common_drug"
                    },
                    {
                        "source": "7",
                        "target": "49",
                        "relation": "common_drug"
                    },
                    {
                        "source": "7",
                        "target": "364",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "7",
                        "target": "356",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "7",
                        "target": "328",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "7",
                        "target": "344",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "7",
                        "target": "345",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "7",
                        "target": "355",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "7",
                        "target": "385",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "7",
                        "target": "173",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "8",
                        "target": "158",
                        "relation": "need_check"
                    },
                    {
                        "source": "8",
                        "target": "143",
                        "relation": "need_check"
                    },
                    {
                        "source": "8",
                        "target": "75",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "8",
                        "target": "79",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "8",
                        "target": "123",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "8",
                        "target": "117",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "8",
                        "target": "77",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "8",
                        "target": "96",
                        "relation": "no_eat"
                    },
                    {
                        "source": "8",
                        "target": "129",
                        "relation": "no_eat"
                    },
                    {
                        "source": "8",
                        "target": "60",
                        "relation": "no_eat"
                    },
                    {
                        "source": "8",
                        "target": "70",
                        "relation": "no_eat"
                    },
                    {
                        "source": "8",
                        "target": "83",
                        "relation": "do_eat"
                    },
                    {
                        "source": "8",
                        "target": "121",
                        "relation": "do_eat"
                    },
                    {
                        "source": "8",
                        "target": "74",
                        "relation": "do_eat"
                    },
                    {
                        "source": "8",
                        "target": "101",
                        "relation": "do_eat"
                    },
                    {
                        "source": "8",
                        "target": "31",
                        "relation": "common_drug"
                    },
                    {
                        "source": "8",
                        "target": "44",
                        "relation": "common_drug"
                    },
                    {
                        "source": "8",
                        "target": "330",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "341",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "352",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "334",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "326",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "336",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "347",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "350",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "369",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "8",
                        "target": "386",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "8",
                        "target": "168",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "9",
                        "target": "167",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "138",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "139",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "144",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "146",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "150",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "134",
                        "relation": "need_check"
                    },
                    {
                        "source": "9",
                        "target": "93",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "57",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "58",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "107",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "102",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "97",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "71",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "128",
                        "relation": "recommand_eat"
                    },
                    {
                        "source": "9",
                        "target": "105",
                        "relation": "no_eat"
                    },
                    {
                        "source": "9",
                        "target": "127",
                        "relation": "no_eat"
                    },
                    {
                        "source": "9",
                        "target": "96",
                        "relation": "no_eat"
                    },
                    {
                        "source": "9",
                        "target": "60",
                        "relation": "no_eat"
                    },
                    {
                        "source": "9",
                        "target": "74",
                        "relation": "do_eat"
                    },
                    {
                        "source": "9",
                        "target": "85",
                        "relation": "do_eat"
                    },
                    {
                        "source": "9",
                        "target": "124",
                        "relation": "do_eat"
                    },
                    {
                        "source": "9",
                        "target": "101",
                        "relation": "do_eat"
                    },
                    {
                        "source": "9",
                        "target": "51",
                        "relation": "common_drug"
                    },
                    {
                        "source": "9",
                        "target": "22",
                        "relation": "common_drug"
                    },
                    {
                        "source": "9",
                        "target": "363",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "363",
                        "relation": "acompany_with"
                    },
                    {
                        "source": "9",
                        "target": "327",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "351",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "340",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "342",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "349",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "336",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "357",
                        "relation": "has_symptom"
                    },
                    {
                        "source": "9",
                        "target": "170",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "173",
                        "target": "171",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "173",
                        "target": "171",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "173",
                        "target": "171",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "173",
                        "target": "171",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "173",
                        "target": "171",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "172",
                        "target": "169",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "172",
                        "target": "169",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "168",
                        "target": "174",
                        "relation": "belongs_to"
                    },
                    {
                        "source": "264",
                        "target": "41",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "195",
                        "target": "37",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "286",
                        "target": "37",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "308",
                        "target": "37",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "261",
                        "target": "37",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "231",
                        "target": "37",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "213",
                        "target": "51",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "213",
                        "target": "51",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "273",
                        "target": "22",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "273",
                        "target": "22",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "235",
                        "target": "22",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "235",
                        "target": "22",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "258",
                        "target": "27",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "237",
                        "target": "39",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "203",
                        "target": "39",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "183",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "198",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "208",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "302",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "47",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "47",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "236",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "225",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "259",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "301",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "216",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "298",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "217",
                        "target": "47",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "305",
                        "target": "50",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "223",
                        "target": "42",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "211",
                        "target": "42",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "275",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "19",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "19",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "209",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "224",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "306",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "196",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "262",
                        "target": "19",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "278",
                        "target": "21",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "282",
                        "target": "49",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "269",
                        "target": "49",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "240",
                        "target": "49",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "322",
                        "target": "49",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "242",
                        "target": "49",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "238",
                        "target": "49",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "271",
                        "target": "31",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "271",
                        "target": "31",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "271",
                        "target": "31",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "271",
                        "target": "31",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "229",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "287",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "274",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "200",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "189",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "315",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "316",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "324",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "210",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "176",
                        "target": "44",
                        "relation": "drugs_of"
                    },
                    {
                        "source": "260",
                        "target": "44",
                        "relation": "drugs_of"
                    }
                ]
            }
            const container = document.getElementById('container');
            const width = container.scrollWidth;
            const height = container.scrollHeight || 500;
            const graph = new G6.Graph({
                container: 'container',
                width,
                height,
                layout: {
                    type: 'force',
                    preventOverlap: true,
                    nodeSize: 20,
                },
                modes: {
                    default: ['zoom-canvas', 'drag-canvas', 'drag-node'],
                },
                defaultNode: {
                    size: 20,
                },
            });

            function getColor1(label) {
                console.log(label)
                if (label == 'Drugs')
                    return '#FED961'
                if (label == 'Foods')
                    return '#91ACE0'
                if (label == 'Checks')
                    return '#ACD78E'
                if (label == 'Departments')
                    return '#E4E1A9'
                if (label == 'Producers')
                    return '#ECA9A1'
                if (label == 'Symptoms')
                    return '#A5CBE2'

                // 返回 RGB 颜色字符串
                else
                    return '#F5B482'
            }


            graph.node(function (node) {
                // console.log(node.c, node.label, node.id)

                return {
                    // label: node.label,
                    labelCfg: {
                        style: {
                            fontSize: 7,
                            // fill: '#aEnergy and Buildingsaa'
                        },
                    },
                    size: 12,
                    style: {
                        fill: getColor1(node.class),
                        stroke: getColor1(node.class),
                        // lineWidth: (node.bold * 3) / 10
                    },
                };
            });

            graph.edge(function (edge) {
                return {
                    label: edge.relation, // 使用 relation 属性作为边的标签
                    labelCfg: {
                        autoRotate: true, // 使标签自动旋转以适应边的方向
                        style: {
                            fontSize: 5,
                            fill: '#aaa',
                        },
                    },
                    style: {
                        stroke: '#ccc',
                        lineWidth: 1,
                    },
                };
            });
            // graph.data(data);
            graph.data({
                nodes: data.nodes.map(node => ({
                    ...node,
                    label: node.name,
                    class: node.label
                })),
                edges: data.edges,
            });
            graph.render();

            function refreshDragedNodePosition(e) {
                const model = e.item.get('model');
                model.fx = e.x;
                model.fy = e.y;
            }
            graph.on('node:dragstart', (e) => {
                graph.layout();
                refreshDragedNodePosition(e);
            });
            graph.on('node:drag', (e) => {
                refreshDragedNodePosition(e);
            });
            if (typeof window !== 'undefined')
                window.onresize = () => {
                    if (!graph || graph.get('destroyed')) return;
                    if (!container || !container.scrollWidth || !container.scrollHeight) return;
                    graph.changeSize(container.scrollWidth, container.scrollHeight);
                };



        }
    }
};
</script>
