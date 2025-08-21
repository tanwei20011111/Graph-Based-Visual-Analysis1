<template>
  <div id="app" class="container">
    <div class="row">
      <div class="box1">
        <div class="title">数据仪表板</div>
        <!-- Data Dashboard -->
        <div class="content">
          <div class="control-panel">
            <div class="subtitle">控制面板</div>
            <!-- Control Panel -->
            <div class="parameter-group">
              <div class="parameter">

                <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
                  <div class="palabel">
                    File Upload:
                  </div>

                </label>


                <div class="custom-file col-sm-8" style="height: 35px;">
                  <input type="file" class="custom-file-input " id="inputGroupFile01"
                    aria-describedby="inputGroupFileAddon01" style="height:35px;">
                  <label class="custom-file-label" for="inputGroupFile01" style=" height: 35px;">Choose
                    Train&Test</label>
                  <!-- <div style="display: inline; width: 10px;height: 10px; opacity: 0;" -->
                  <!-- @click="initdata">1</div> -->
                </div>


              </div>


              <div class="parameter">

                <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
                  <div class="palabel">
                    Model:
                  </div>
                  &ensp;&ensp;&ensp;&ensp;
                </label>
                <select class="custom-select col-sm-8" id="inlineFormCustomSelectPref" style="height: 35px;">
                  <option selected>GCN + graphLSTM</option>
                  <option value="1">GAT + graphLSTM</option>
                  <option value="2">graphLSTM</option>
                  <option value="3">LSTM</option>
                  <option value="4">VAE</option>
                  <option value="5">GAN</option>
                </select>
              </div>
              <div class="parameter">

                <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
                  <div class="palabel">
                    Window:
                  </div>
                  &ensp;
                </label>
                <select class="custom-select col-sm-2" id="inlineFormCustomSelectPref" style="height: 35px;">
                  <option value="30"> 30</option>
                  <option value="5">5</option>
                  <option value="10" selected>10</option>
                  <option value="20">20</option>
                  <option value="30">30</option>
                  <option value="50">50</option>
                </select>
                <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
                  <div class="palabel">
                    &ensp;Threshold:
                  </div>

                </label>
                <select class="custom-select col-sm-3" id="inlineFormCustomSelectPref" style="height: 35px;">
                  <option value="1" selected> Mean</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                  <option value="5">5</option>
                  <option value="6">6</option>
                </select>

                <!-- <div class="form-check form-check-inline">
                  <input class="form-check-input" type="radio" name="inlineRadioOptions" id="inlineRadio1"
                    value="option1" checked>
                  <label class="form-check-label" for="inlineRadio1">1</label>
                </div>
                <div class="form-check form-check-inline">
                  <input class="form-check-input" type="radio" name="inlineRadioOptions" id="inlineRadio2"
                    value="option2">
                  <label class="form-check-label" for="inlineRadio2">2</label>
                </div>
                <div class="form-check form-check-inline">
                  <input class="form-check-input" type="radio" name="inlineRadioOptions" id="inlineRadio3"
                    value="option3">
                  <label class="form-check-label" for="inlineRadio3">3</label>
                </div> -->
              </div>


              <div class="parameter">

                <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
                  <div class="palabel">
                    Epochs:
                  </div>

                </label>
                <!-- <select class="custom-select col-sm-3" id="inlineFormCustomSelectPref" style="height: 35px;">
                  <option value="1" selected> Mean</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                  <option value="5">5</option>
                  <option value="6">6</option>
                </select> -->
                <!-- <input type="range" min="1" max="100" value="50" style="width: 39%;" color="grey"> -->
                &ensp;&ensp;<input class="form-control form-control-sm col-sm-2" type="text" placeholder="50"
                  style="display: inline;">
                &ensp;
                <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
                  <div class="palabel">
                    Learning rate:
                  </div>

                </label>
                <!-- <select class="custom-select col-sm-3" id="inlineFormCustomSelectPref" style="height: 35px;">
                  <option value="1" selected> Mean</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                  <option value="5">5</option>
                  <option value="6">6</option>
                </select> -->
                <!-- <input type="range" min="1" max="100" value="50" style="width: 39%;" color="grey"> -->
                <input class="form-control form-control-sm col-sm-2" type="text" placeholder="0.001"
                  style="display: inline;">

              </div>
              <!-- @click="hPartition" -->
              <div style="margin-top: 2px; position: relative; left: 33%;">
                <button type="button" class="btn btn-outline-secondary"
                  style="border-color: #aaa ;font-size: 15px;">Start
                  Analysis</button>
              </div>

            </div>
            <hr />
          </div>
          <div class="time-series-chart">
            <div class="subtitle">数据集预览</div>
            <!-- Dataset Info -->
            <div class="palabel">
              Time Seris:&ensp;590
            </div>
            <!-- <input class="form-control form-control-sm col-sm-2" type="text" style="display: inline;"
              :value="dataPoints"> -->
            &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
            <div class="palabel">
              Metrics:&ensp;62
            </div>
            <!-- <input class="form-control form-control-sm col-sm-2" type="text" style="display: inline;" :value="metrics">
            <div style="height: 10px;"></div> -->
            <br>
            <div class="palabel">
              Time Span:&ensp;531 min
            </div>
            <!-- <input class="form-control form-control-sm col-sm-2" type="text" style="display: inline;" :value="timeSpan"> -->
            &ensp;&ensp;
            <div class="palabel">
              Time Interval:&ensp;54 s
            </div>
            <!-- <input class="form-control form-control-sm col-sm-2" type="text" style="display: inline;"
              :value="timeInterval"> -->


            <div class="chart-placeholder" style="height: 423px;width: 102%;  overflow: scroll;">
              <!-- 在这里添加折线图组件 -->
              <div id="tm1" style="width: 94%;height: 195px;  position: relative;bottom: 30px;"></div>
              <div id="tm2" style="width: 100%;height: 210px;position: relative;bottom: 70px;"></div>
              <div id="tm3" style="width: 100%;height: 220px; position: relative;bottom: 140px;"></div>
            </div>
          </div>
        </div>
        <!-- 在这里添加你的图表组件 -->
      </div>
      <div class="box2">
        <div class="title">异常检测</div>
        <!-- Abnormal Detection -->
        <div class="nav">
          <div class="s" style="width:350px; overflow: auto;">
            <div id="sandian" style="width:380px; height: 47px; position: absolute;left: 8px; z-index: 15;">

            </div>

          </div>
          <div style="position: absolute;left: 408px;top: 34px;">
            <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
              <div class="palabel">
                Threshold:
              </div>

            </label>
            <input type="range" min="1" max="100" value="50" style="width: 38%;" color="grey">
          </div>
          <div style="position: absolute;width: 180px;left: 605px;top: 34px;">
            <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
              <div class="palabel">
                Metrics:
              </div>

            </label>
            <select class="custom-select col-sm-4" id="inlineFormCustomSelectPref"
              style="height: 30px;font-size: small;">
              <option value="30" selected> 6</option>
              <option value="5">5</option>
              <option value="10">4</option>
              <option value="20">7</option>
              <option value="30">8</option>
              <option value="50">9</option>
            </select>
          </div>
          <div class="palabel" style="position: absolute;right: 190px;top: 40px;">Abnormal: </div>
          <div class="red-circle" style="position: absolute;right: 175px;top: 46px;"></div>
          <div class="palabel" style="position: absolute;right: 295px;top: 40px;">Normal: </div>
          <div class="grey-circle" style="position: absolute;right: 280px;top: 46px;"></div>
          <div style="position: absolute; top: 30px;right: 20px;"><img src="../../static/img/img1.png" alt=""
              style="width: 140px;"></div>

        </div>
        <div class="con" style="overflow: scroll; width: 1110px; position: relative; bottom: 5px;">
          <div class="huihe" id="relizhexian" style="width: 7910px; height: 210px;z-index: 10;z-index: 10;">

          </div>

        </div>
        <!-- 在这里添加你的图表组件 -->
      </div>
    </div>
    <div class="row">
      <div class="box3">
        <div class="title">异常分析</div>
<!-- Anomaly Analysis -->
        <div class="nav2">
          <div class="left1" style="position: absolute;left: 10px;top: 35px; width: 250px;">
            <div class="palabel">
              Layout:
            </div>
            <select class="custom-select col-sm-6" id="inlineFormCustomSelectPref"
              style="height: 30px;font-size: small;z-index: 15;">
              <option value="30" selected> Circular</option>
              <option value="5">Force-directed</option>
              <option value="10">4</option>
              <option value="20">7</option>
              <option value="30">8</option>
              <option value="50">9</option>
            </select>
          </div>

          <div class="left2" style="position: absolute;left: 210px;top: 35px; width: 250px;">
            <div class="palabel">
              Edge Bundling:
            </div>
            <input type="checkbox" class="form-check-input" id="exampleCheck1"
              style="position: relative;left: 23px;top: 3px;" checked>
          </div>

          <div class="left3" style="position: absolute;left: 350px;top: 35px; width: 250px;">
            <div class="palabel">
              Highlight:
            </div>
            <input type="checkbox" class="form-check-input" id="exampleCheck1"
              style="position: relative;left: 23px;top: 3px;">
          </div>
          <div style="position: absolute;left: 455px;top: 33px;">
            <label class="my-1 mr-2" for="inlineFormCustomSelectPref">
              <div class="palabel">
                Threshold:
              </div>

            </label>
            <input type="range" min="1" max="100" value="50" style="width: 38%;" color="grey">
          </div>


          <div class="left3" style="position: absolute;left: 640px;top: 35px; width: 150px;">
            <div class="palabel">
              Radius:
            </div>
            <select class="custom-select col-sm-4" id="inlineFormCustomSelectPref"
              style="height: 30px;font-size: small;z-index: 15;">
              <option value="30" selected> 2</option>
              <option value="5">3</option>
              <option value="10">4</option>

            </select>
          </div>



          <div style="position: absolute;left: 945px; top: 38px;">
            <div class="palabel">
              Prediction:&ensp;
            </div><img src="../../static/img/node_predict1.png" alt="">
          </div>
          <div style="position: absolute;left: 780px; top: 38px;">
            <div class="palabel">
              Observation:&ensp;
            </div><img src="../../static/img/node_obser.png" alt="">
          </div>
        </div>
        <!-- 在这里添加你的图表组件 -->
        <div class="graphvis" style="width: 100%;height: 80%">

          <div id="container"
            style="width:260px;height: 250px;border: 0.5px solid #ccc; position: relative;top: 45px;left: 133px;">
          </div>
          <div id="container2"
            style="width:260px;height: 250px;border: 0.5px solid #ccc;z-index: 5; position: relative;bottom:205px;left: 395px;">
          </div>

          <div style="position: absolute;top:73px;left: 5px; width: 128px;height: 251px;overflow: auto;">
            <div id="container3" style="width:120px;height: 120px;border: 0.5px solid #ccc; position: relative;">
            </div>
            <div id="container4" style="width:120px;height: 120px;border: 2px solid #ccc; position: relative;top:5px;">
            </div>
            <div id="container5"
              style="width:120px;height: 120px;border: 2px solid #D98574; position: relative;top:10px;">
            </div>
            <div id="container6"
              style="width:120px;height: 120px;border: 0.5px solid #D98574; position: relative;top:15px;">
            </div>
            <div id="container7"
              style="width:120px;height: 120px;border: 0.5px solid #D98574; position: relative;top:20px;">
            </div>
          </div>


          <div style="position: relative; bottom:455px; left: 656px;width: 450px;height: 253px;overflow: scroll;">
            <div class="metrics"
              style="width:220px;height: 121px;border: 0.5px solid #ccc; position: relative;left: 1px; overflow: auto;">
              <div id="zhexianx" style="width: 100%;height: 110%;"></div>
            </div>
            <div class="metrics"
              style="width:220px;height: 121px;border: 0.5px solid #ccc; position: relative;left: 220px;bottom: 121px; overflow: auto;">
              <div id="zhexianx2" style="width: 100%;height: 110%;"></div>
            </div>
            <div class="metrics"
              style="width:220px;height: 121px;border: 0.5px solid #ccc; position: absolute;top:120px;left:1px; overflow: auto;">
              <div id="zhexianx3" style="width: 100%;height: 110%;"></div>
            </div>
            <div class="metrics"
              style="width:220px;height: 121px;border: 0.5px solid #ccc; position: relative;bottom:122px;left: 220px; overflow: auto;">
              <div id="zhexianx4" style="width: 100%;height: 110%;"></div>
            </div>
            <div class="metrics"
              style="width:220px;height: 121px;border: 0.5px solid #ccc; position: relative; overflow: auto;">
              <div id="zhexianx5" style="width: 100%;height: 110%;"></div>
            </div>

          </div>



        </div>
      </div>
    </div>
    <div class="row">
      <div class="box4">
        <div class="title">异常解释</div>
<!-- LLM Explanation -->


        <div class="form-group" style="margin-left: 10px;margin-top: 8px; width: 30%;">
          <label for="exampleFormControlTextarea1" class="palabel" style="color: #2E54A1;">Prompt Template</label>
          <button type="button" class="btn btn-light" style="position: absolute;top: 30px;left:  270px;">Save</button>
          <button type="button" class="btn btn-light"
            style="position: absolute;top: 30px;left:  630px;z-index: 10;;">Submit</button>
          <button type="button" class="btn btn-light"
            style="position: absolute;top: 30px;left:  1020px;z-index: 10;;">Copy</button>
          <!-- <textarea class="form-control col-sm-3" id="exampleFormControlTextarea1" rows="6"></textarea> -->

          <!-- <ul class="nav nav-tabs" style="height: 35px;">
            <li class="nav-item">
              <a class="nav-link" :class="{ 'active': currentTab === 'tab1' }" @click="funnav('tab1')">Instruction</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" :class="{ 'active': currentTab === 'tab2' }" @click="funnav('tab2')">Context</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" :class="{ 'active': currentTab === 'tab3' }" @click="funnav('tab3')">Input</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" :class="{ 'active': currentTab === 'tab4' }" @click="funnav('tab4')">Output</a>
            </li>
          </ul> -->
          <ul class="list-group" style="position: relative;top: 11px;">
            <li class="list-group-item" @dblclick="funnav"> <span v-if="!showTextarea">&lt;Instruction&gt;Analyze
                the given HVAC syste</span>
              <textarea v-else class="form-control" id="exampleFormControlTextarea1" rows="4">
Analyze the given HVAC system energy consumption data to identify the cause of anomalies and recommend solutions. Focus on the most anomalous metrics and their surrounding time points.
      </textarea>
            </li>
            <li class="list-group-item">&lt;Context&gt;The HVAC dataset was simulated</li>
            <li class="list-group-item">&lt;Input Data&gt;The following are the most an</li>
            <li class="list-group-item">&lt;Output Indicator&gt;The answer is what abn </li>



          </ul>
          <!-- <div>
            <button @click="currentTab = 'tab1'">Tab 1</button>
            <button @click="currentTab = 'tab2'">Tab 2</button>
            <button @click="currentTab = 'tab3'">Tab 3</button>
          </div> -->

          <!-- 文本区域 -->
          <!-- <div class="content1" style="width: 100%;margin: 0;">
            <div v-if="currentTab === 'tab1'">
              <textarea class="form-control" id="exampleFormControlTextarea1" rows="6">
              Analyze the given HVAC system energy consumption data to identify the cause of anomalies and recommend solutions. Focus on the most anomalous metrics and their surrounding time points.
              </textarea>
            </div>
            <div v-if="currentTab === 'tab2'"><textarea class="form-control" id="exampleFormControlTextarea1"
                rows="6">2</textarea></div>
            <div v-if="currentTab === 'tab3'"><textarea class="form-control" id="exampleFormControlTextarea1"
                rows="6">3</textarea></div>
            <div v-if="currentTab === 'tab4'"><textarea class="form-control" id="exampleFormControlTextarea1"
                rows="6"></textarea></div>
          </div> -->

        </div>

        <div class="form-group" style="position: relative; bottom: 222px;left: 350px;">
          <label for="exampleFormControlTextarea1" class="palabel" @click="setRows"
            style="color: #2E54A1;">Preview</label>

          <textarea class="form-control col-sm-4" id="exampleFormControlTextarea1" :rows="rows"
            style="position: relative; top: 11px;">Instruction:
Analyze the given HVAC system energy consumption data to identify the cause of anomalies and recommend solutions. Focus on the most anomalous metrics and their surrounding time points.

Context:
The HVAC dataset was simulated using the Transient System Simulation tool, including a three-story office building A,B,C. Air handling units are installed on each floor, and variable air volume terminals control the variable flow. The water tank in this HVAC system uses a flow pump to supply chilled water to the cooling coils. In total, the dataset simulates a twelve-zone HVAC dataset, which includes the measured values of temperature sensors, control signals, temperature set points, residential thermal comfort in each zone, the total estimated power, and the status of the system.
Metric Explanation of symbols： T: temperature, sensor_T: sensor temperature reading

Input:
The following are the most anomalous metrics and their values at the time of the anomaly, along with their values two time points before and after the anomaly:

 [{  Metric: Power
   Time t-2: 556389.9533
   Time t-1: 572200.1444
   Time t: 229417.0531
   Time t+1: 294624.633
   Time t+2: 353850.7342
 }

 {  Metric: T_chiller
   Time t-2: 9°C
   Time t-1: 9°C
   Time t: 14°C (anomalous)
   Time t+1: 14°C
   Time t+2: 14°C
 }
 {  Metric: sensor_T_chiller
   Time t-2: 9.007°C
   Time t-1: 9.007°C
   Time t: 11.622°C (anomalous)
   Time t+1: 12.507°C
   Time t+2: 13.601°C
}]

Output:
The answer is what abnormality has occurred in the HVAC system at this moment, the cause, and the solution.
Format:
Answer:
abnormal:
Possible Cause: Briefly explain it in one sentence.
Recommendation: Briefly explain it in one sentence.</textarea>
        </div>

        <div class="form-group" style="position: absolute; top: 37px;left: 725px; width: 34%;">
          <label for="exampleFormControlTextarea1" class="palabel" style="color: #C81D31;">Response</label>
          <textarea class="form-control" id="exampleFormControlTextarea1" rows="7"
            style="position: relative; top: 9px;">Answer:
abnormal: A sudden drop in power consumption and a spike in chiller temperature.
Possible Cause: The chiller's cooling efficiency has drastically reduced, possibly due to a malfunction or a failure in the cooling system.
Recommendation: Inspect and repair the chiller unit and associated components, ensuring proper coolant levels and checking for any mechanical issues.</textarea>
        </div>



        <!-- 在这里添加你的图表组件 -->
      </div>
    </div>
  </div>
</template>

<script>
import G6 from '@antv/g6';
import axios from 'axios';
import * as echarts from 'echarts';


export default {
  name: 'HelloWorld',
  data() {
    return {
      threshold: 0,
      window: 0,
      slideStep: 50,
      dataPoints: 10000,
      metrics: 62,
      timeSpan: '24 h',
      timeInterval: '1 s',
      currentTab: 'tab1',
      rows: 7,
      graph1: [],
      flag: true,
      showTextarea: false,

    }
  },
  mounted() {

    this.gettimeSeris()
    this.showhuihe()
    this.showg1()
    this.showg3()
    this.showmetriczhexian()
    this.showmetriczhexian2()
    this.showmetriczhexian3()
    this.showmetriczhexian4()
    this.showsandian()
  },
  methods: {
    showsandian() {
      var chartDom = document.getElementById('sandian');
      var myChart = echarts.init(chartDom);
      var option;

      // prettier-ignore
      const hours = ['09:44:18', '09:59:29', '10:14:40', '10:52:12',
        '', '', '', '',
        '', '', '', '',
        '', '', '',
        '12:03:18', '', '', '',
        '', '', '', '12:46:30'];
      // prettier-ignore
      const days = [
        '1'
      ];
      // prettier-ignore
      const data = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 2], [0, 4, 0], [0, 5, 0], [0, 6, 0], [0, 7, 0], [0, 8, 0], [0, 9, 0], [0, 10, 0], [0, 11, 0], [0, 12, 0], [0, 13, 0], [0, 14, 0], [0, 15, 7], [0, 16, 7], [0, 17, 6], [0, 18, 4], [0, 19, 4], [0, 20, 3], [0, 21, 4], [0, 22, 3], [0, 22.5, 2], [0, 23, 0], [0, 24, 0], [0, 25, 0]];
      const title = [];
      const singleAxis = [];
      const series = [];
      days.forEach(function (day, idx) {
        title.push({
          textBaseline: 'middle',
          top: ((idx + 0.5) * 100) / 7 + '%',
          text: day,
          show: false
        });
        singleAxis.push({
          left: 22,
          type: 'category',
          boundaryGap: false,
          data: hours,
          top: '20%',
          height: '25%',
          axisLabel: {
            interval: 2
          }
        });
        series.push({
          singleAxisIndex: idx,
          coordinateSystem: 'singleAxis',
          type: 'scatter',
          data: [],
          symbolSize: function (dataItem) {
            return dataItem[1] * 4;
          },
          itemStyle: {
            color: '#D98574' // 设置气泡颜色为红色
          }

        });
      });
      data.forEach(function (dataItem) {
        series[dataItem[0]].data.push([dataItem[1], dataItem[2]]);
      });
      option = {
        tooltip: {
          position: 'top'
        },
        title: title,
        singleAxis: singleAxis,
        series: series
      };

      option && myChart.setOption(option);
    },
    gettimeSeris() {
      const path = 'http://127.0.0.1:5000/getrawdata';
      const msg1 = axios.get(path)
        .then((res) => {
          console.log(res.data)


          this.showTimeSeris1(res.data.x, res.data.t1, res.data.t1_name)
          this.showTimeSeris2(res.data.x, res.data.t2, res.data.t2_name)
          this.showTimeSeris3(res.data.x, res.data.t3, res.data.t3_name)


        })
        .catch((error) => {
          console.error(error);
        });
      return msg1

    },

    setRows() {

      if (this.flag == true) { this.rows = 20; }
      else this.rows = 7;
      this.flag = !this.flag
      // 点击label后设置为12行
    },
    funnav() {
      this.showTextarea = !this.showTextarea
    },

    handleFileUpload(event) {
      const file = event.target.files[0];
      // 在这里处理文件上传
      console.log(file);
    },
    showmetriczhexian() {
      var chartDom = document.getElementById('zhexianx');
      var myChart = echarts.init(chartDom);
      var option;

      option = {
        title: {
          text: 'Setpoints_Chiller_T_chiller',
          textStyle: {
            fontSize: 12  // Adjust the font size as needed
          },
          top: '15%'
        },
        color: ['#EED399', '#D88273'],
        tooltip: {
          trigger: 'axis'
        },
        // legend: {
        //   data: ['Observation', 'Prediction']
        // },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '1%',
          containLabel: true
        },
        toolbox: {
          feature: {
            saveAsImage: {}
          },
          top: '5%'
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: [
            '12:01:30', '12:02:24', '12:03:18', '12:04:12', '12:05:06'
          ]
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 20
        },
        series: [
          {
            name: 'Observation',
            type: 'line',
            data: [9, 9, 14, 14, 14],

          },
          {
            name: 'Prediction',
            type: 'line',
            data: [
              8.968167, 8.899981, 8.926297, 9.1779175,
              9.331415
            ]
          }
        ]
      };

      option && myChart.setOption(option);

    },
    showmetriczhexian2() {
      var chartDom = document.getElementById('zhexianx3');
      var myChart = echarts.init(chartDom);
      var option;

      option = {
        title: {
          text: 'power',
          textStyle: {
            fontSize: 12  // Adjust the font size as needed
          },
          top: '15%'
        },
        color: ['#EED399', '#D88273'],
        tooltip: {
          trigger: 'axis'
        },
        // legend: {
        //   data: ['Observation', 'Prediction']
        // },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '1%',
          containLabel: true
        },
        toolbox: {
          feature: {
            saveAsImage: {}
          },
          top: '5%'
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: [
            '12:01:30', '12:02:24', '12:03:18', '12:04:12', '12:05:06'
          ]
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 10
        },
        series: [
          {
            name: 'Observation',
            type: 'line',
            data: [5.56389953285103, 5.722001443686371, 2.29417053112663, 2.94624632974919, 3.5385073421887907]
          },
          {
            name: 'Prediction',
            type: 'line',
            data: [5.3871200561523445, 5.4823852539062505, 5.432783508300782, 5.937192153930664, 6.233164978027344]
          }
        ]
      };

      option && myChart.setOption(option);

    },
    showmetriczhexian3() {
      var chartDom = document.getElementById('zhexianx2');
      var myChart = echarts.init(chartDom);
      var option;

      option = {
        title: {
          text: 'Sensor_Chiller_T_chiller',
          textStyle: {
            fontSize: 12  // Adjust the font size as needed
          },
          top: '15%'
        },
        color: ['#EED399', '#D88273'],
        tooltip: {
          trigger: 'axis'
        },
        // legend: {
        //   data: ['Observation', 'Prediction']
        // },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '1%',
          containLabel: true
        },
        toolbox: {
          feature: {
            saveAsImage: {}
          },
          top: '5%'
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: [
            '12:01:30', '12:02:24', '12:03:18', '12:04:12', '12:05:06'
          ]
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 20
        },
        series: [
          {
            name: 'Observation',
            type: 'line',
            data: [8.986, 8.998, 11.622, 12.507, 13.146]
          },
          {
            name: 'Prediction',
            type: 'line',
            data: [
              8.975148, 8.958384, 8.91704, 8.941872, 9.299278
            ]
          }
        ]
      };

      option && myChart.setOption(option);

    },
    showmetriczhexian4() {
      var chartDom = document.getElementById('zhexianx4');
      var myChart = echarts.init(chartDom);
      var option;

      option = {
        title: {
          text: 'Sensor_AHU_T_woA',
          textStyle: {
            fontSize: 12  // Adjust the font size as needed
          },
          top: '15%'
        },
        tooltip: {
          trigger: 'axis'
        },
        color: ['#EED399', '#D88273'],
        // legend: {
        //   data: ['Observation', 'Prediction']
        // },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '1%',
          containLabel: true
        },
        toolbox: {
          feature: {
            saveAsImage: {}
          },
          top: '5%'
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: [
            '12:01:30', '12:02:24', '12:03:18', '12:04:12', '12:05:06'
          ]
        },
        yAxis: {
          type: 'value',
          min: 10,
          max: 20
        },
        series: [
          {
            name: 'Observation',
            type: 'line',
            data: [16.555, 16.58, 17.175, 17.621, 17.926]
          },
          {
            name: 'Prediction',
            type: 'line',
            data: [15.911948, 15.986797, 15.927169, 16.669596, 16.714511]
          }
        ]
      };

      option && myChart.setOption(option);

    },
    showsmallgraph(data, id) {

      const tooltip = new G6.Tooltip({
        offsetX: 20,
        offsetY: -40,
        // the types of items that allow the tooltip show up
        // 允许出现 tooltip 的 item 类型
        itemTypes: ['node'],
        // custom the tooltip's content
        // 自定义 tooltip 内容
        getContent: (e) => {
          const outDiv = document.createElement('div');
          outDiv.style.width = 'fit-content';
          outDiv.style.padding = '2px 2px 2px 2px';
          outDiv.innerHTML = `
          ${e.item.getModel().label}`;
          return outDiv;
        },
      });


      function getColor1(value) {
        var min = 0
        var max = 5
        // 确保数值在 [min, max] 范围内
        value = Math.max(min, Math.min(max, value));

        // 线性插值计算比例
        let ratio = (value - min) / (max - min);

        // 定义颜色范围，这里以蓝色到红色渐变为例
        let startColor = { r: 245, g: 236, b: 167 }; // 蓝色
        let endColor = { r: 196, g: 68, b: 79 }; // 红色

        // 计算当前颜色值
        let r = Math.round(startColor.r + ratio * (endColor.r - startColor.r));
        let g = Math.round(startColor.g + ratio * (endColor.g - startColor.g));
        let b = Math.round(startColor.b + ratio * (endColor.b - startColor.b));

        // 返回 RGB 颜色字符串
        return `rgb(${r}, ${g}, ${b})`;
      }



      const edgeBundling = new G6.Bundling({
        bundleThreshold: 0.4,
      });
      const container = document.getElementById(id);
      const width = container.scrollWidth;
      const height = container.scrollHeight || 500;
      const graph = new G6.Graph({
        container: id,
        width,
        height,

        layout: {
          type: 'force',
          preventOverlap: true,
          center: [width / 2, height / 2],
          // linkDistance: 100,  // 调整连线距离
          //       nodeStrength: -30,  // 调整节点之间的引力
          //       edgeStrength: 0.1,  // 调整边的弹力
        },
        plugins: [edgeBundling],
        modes: {
          default: ['zoom-canvas', 'drag-canvas', 'drag-node'],
        },
        defaultNode: {
          size: 5,
        },
      });
      graph.node(function (node) {
        // console.log(node.c, node.label, node.id)

        return {
          label: node.label,
          labelCfg: {
            style: {
              opacity: 0,
            },
          },
          size: 5,
          style: {
            fill: getColor1(node.c),
            stroke: getColor1(node.c),
            // lineWidth: (node.bold * 3) / 10
          },
        };
      });
      // graph.edge(function (edge) {
      //   return {
      //     style: {
      //       // stroke: edge.newadd,
      //       lineWidth: edge.newadd
      //     }
      //   }
      // });

      graph.data(data);
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

      // setTimeout(() => {
      //   edgeBundling.bundling(data);
      // }, 1000);

      if (typeof window !== 'undefined')
        window.onresize = () => {
          if (!graph || graph.get('destroyed')) return;
          if (!container || !container.scrollWidth || !container.scrollHeight) return;
          graph.changeSize(container.scrollWidth, container.scrollHeight);
          // logNodePositions();
        };

    },
    //showg1 有节点边设计 使用getgraph3后能指定坐标 使用getGraph1就是力导向布局
    showgraph(data, id) {

      const tooltip = new G6.Tooltip({
        offsetX: 20,
        offsetY: -40,
        // the types of items that allow the tooltip show up
        // 允许出现 tooltip 的 item 类型
        itemTypes: ['node'],
        // custom the tooltip's content
        // 自定义 tooltip 内容
        getContent: (e) => {
          const outDiv = document.createElement('div');
          outDiv.style.width = 'fit-content';
          outDiv.style.padding = '2px 2px 2px 2px';
          outDiv.innerHTML = `
          ${e.item.getModel().label}`;
          return outDiv;
        },
      });


      function getColor1(value) {
        var min = 0
        var max = 5
        // 确保数值在 [min, max] 范围内
        value = Math.max(min, Math.min(max, value));

        // 线性插值计算比例
        let ratio = (value - min) / (max - min);

        // 定义颜色范围，这里以蓝色到红色渐变为例
        let startColor = { r: 245, g: 236, b: 167 }; // 蓝色
        let endColor = { r: 196, g: 68, b: 79 }; // 红色

        // 计算当前颜色值
        let r = Math.round(startColor.r + ratio * (endColor.r - startColor.r));
        let g = Math.round(startColor.g + ratio * (endColor.g - startColor.g));
        let b = Math.round(startColor.b + ratio * (endColor.b - startColor.b));

        // 返回 RGB 颜色字符串
        return `rgb(${r}, ${g}, ${b})`;
      }



      const edgeBundling = new G6.Bundling({
        bundleThreshold: 0.6,
      });
      const container = document.getElementById(id);
      const width = container.scrollWidth;
      const height = container.scrollHeight || 500;
      const graph = new G6.Graph({
        container: id,
        width,
        height,

        layout: {
          type: 'force',
          preventOverlap: true,
          center: [width / 2, height / 2],
          // linkDistance: 100,  // 调整连线距离
          //       nodeStrength: -30,  // 调整节点之间的引力
          //       edgeStrength: 0.1,  // 调整边的弹力
        },
        plugins: [tooltip, edgeBundling],
        modes: {
          default: ['zoom-canvas', 'drag-canvas', 'drag-node'],
        },
        defaultNode: {
          size: 10,
        },
      });
      graph.node(function (node) {
        // console.log(node.c, node.label, node.id)

        return {
          label: node.label,
          labelCfg: {
            style: {
              opacity: 0,
            },
          },
          size: 12,
          style: {
            fill: getColor1(node.c),
            stroke: node.bolder,
            lineWidth: (node.bold * 3) / 10
          },
        };
      });
      // graph.edge(function (edge) {
      //   return {
      //     style: {
      //       stroke: edge.newadd,
      //       // lineWidth: edge.newadd
      //     }
      //   }
      // });

      graph.data(data);
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

      setTimeout(() => {
        edgeBundling.bundling(data);
      }, 1000);

      if (typeof window !== 'undefined')
        window.onresize = () => {
          if (!graph || graph.get('destroyed')) return;
          if (!container || !container.scrollWidth || !container.scrollHeight) return;
          graph.changeSize(container.scrollWidth, container.scrollHeight);
          // logNodePositions();
        };

    },
    //大图获取后  show2中 使用环形/力导向 保存json坐标 代码可变化 没节点边设计
    showgraph2(data, id) {



      function getColor1(value) {
        var min = 0
        var max = 5
        // 确保数值在 [min, max] 范围内
        value = Math.max(min, Math.min(max, value));

        // 线性插值计算比例
        let ratio = (value - min) / (max - min);

        // 定义颜色范围，这里以蓝色到红色渐变为例
        let startColor = { r: 245, g: 236, b: 167 }; // 蓝色
        let endColor = { r: 196, g: 68, b: 79 }; // 红色

        // 计算当前颜色值
        let r = Math.round(startColor.r + ratio * (endColor.r - startColor.r));
        let g = Math.round(startColor.g + ratio * (endColor.g - startColor.g));
        let b = Math.round(startColor.b + ratio * (endColor.b - startColor.b));

        // 返回 RGB 颜色字符串
        return `rgb(${r}, ${g}, ${b})`;
      }


      const container = document.getElementById(id);
      const width = container.scrollWidth + 5;
      const height = container.scrollHeight;


      const tooltip = new G6.Tooltip({
        offsetX: 8,
        offsetY: 8,
        // the types of items that allow the tooltip show up
        // 允许出现 tooltip 的 item 类型
        itemTypes: ['node'],
        // custom the tooltip's content
        // 自定义 tooltip 内容
        getContent: (e) => {
          const outDiv = document.createElement('div');
          outDiv.style.width = 'fit-content';
          outDiv.style.padding = '2px 2px 2px 2px';
          outDiv.innerHTML = `
          ${e.item.getModel().label}`;
          return outDiv;
        },
      });
      const graph = new G6.Graph({
        container: id,
        width,
        height,
        modes: {
          default: ['drag-canvas', 'drag-node'],
        },
        plugins: [tooltip],
        layout: {
          type: 'circular',
          radius: 50,
        },
        animate: true,
        // defaultNode: {
        //   size: 10,
        //   style: {
        //     fill: '#ccc',
        //     stroke: 'red',
        //     lineWidth: 1,
        //   },
        // },
      });



      graph.node(function (node) {
        console.log(node.c, node.label, node.id)

        return {
          label: node.label,
          labelCfg: {
            style: {
              opacity: 0,
            },
          },
          size: 5,
          style: {
            fill: getColor1(node.c),
            stroke: getColor1(node.c),
            // lineWidth: (node.bold * 3) / 10
          },
        };
      });

      // graph.edge(function (edge) {
      //   return {
      //     style: {
      //       // stroke: edge.newadd,
      //       lineWidth: edge.newadd
      //     }
      //   }
      // });
      graph.data(data);
      graph.render();

      if (typeof window !== 'undefined')
        window.onresize = () => {
          if (!graph || graph.get('destroyed')) return;
          if (!container || !container.scrollWidth || !container.scrollHeight) return;
          graph.changeSize(container.scrollWidth, container.scrollHeight);
        };



      function logNodePositions() {
        const nodes = graph.getNodes();
        const nodeList = [];

        nodes.forEach((node) => {
          const model = node.getModel();
          const nodeData = {
            id: model.id,
            position: [model.x, model.y]
          };
          nodeList.push(nodeData);
        });

        // Convert the list to a JSON string
        const jsonString = JSON.stringify(nodeList, null, 2);
        console.log(jsonString)

        // Save the JSON string to a file
        saveJsonToFile(jsonString, 'nodes_cir_all.json');
      }
      function saveJsonToFile(jsonString, filename) {
        // Create a blob from the JSON string
        const blob = new Blob([jsonString], { type: 'application/json' });

        // Create a URL for the blob
        const url = URL.createObjectURL(blob);

        // Create a link element
        const link = document.createElement('a');

        // Set the download attribute with the filename
        link.download = filename;

        // Set the href attribute to the blob URL
        link.href = url;

        // Programmatically click the link to trigger the download
        link.click();

        // Clean up by revoking the object URL
        URL.revokeObjectURL(url);
      }

      // Call this function whenever you need to log the positions
      setTimeout(() => {
        logNodePositions();
      }, 6000)




    },
    //showg1 有节点边设计 不能使用getgraph3指定坐标 使用getGraph1就是环形
    showgraph_cir(data, id) {

      const container = document.getElementById(id);
      const width = container.scrollWidth + 5;
      const height = container.scrollHeight;


      const tooltip = new G6.Tooltip({
        offsetX: 8,
        offsetY: 8,
        // the types of items that allow the tooltip show up
        // 允许出现 tooltip 的 item 类型
        itemTypes: ['node'],
        // custom the tooltip's content
        // 自定义 tooltip 内容
        getContent: (e) => {
          const outDiv = document.createElement('div');
          outDiv.style.width = 'fit-content';
          outDiv.style.padding = '2px 2px 2px 2px';
          outDiv.innerHTML = `
          ${e.item.getModel().label}`;
          return outDiv;
        },
      });
      const graph = new G6.Graph({
        container: id,
        width,
        height,
        modes: {
          default: ['drag-canvas', 'drag-node'],
        },
        plugins: [tooltip],
        layout: {
          type: 'circular',
          radius: 110,
        },
        animate: true,
        // defaultNode: {
        //   size: 10,
        //   style: {
        //     fill: '#ccc',
        //     stroke: 'red',
        //     lineWidth: 1,
        //   },
        // },
      });

      function getColor1(value) {
        var min = 0
        var max = 5
        // 确保数值在 [min, max] 范围内
        value = Math.max(min, Math.min(max, value));

        // 线性插值计算比例
        let ratio = (value - min) / (max - min);

        // 定义颜色范围，这里以蓝色到红色渐变为例
        let startColor = { r: 245, g: 236, b: 167 }; // 蓝色
        let endColor = { r: 196, g: 68, b: 79 }; // 红色

        // 计算当前颜色值
        let r = Math.round(startColor.r + ratio * (endColor.r - startColor.r));
        let g = Math.round(startColor.g + ratio * (endColor.g - startColor.g));
        let b = Math.round(startColor.b + ratio * (endColor.b - startColor.b));

        // 返回 RGB 颜色字符串
        return `rgb(${r}, ${g}, ${b})`;
      }


      graph.node(function (node) {
        console.log(node.c, node.label, node.id)

        return {
          label: node.label,
          labelCfg: {
            style: {
              opacity: 0,
            },
          },
          size: 12,
          style: {
            fill: getColor1(node.c),
            stroke: node.bolder,
            lineWidth: (node.bold * 3) / 10
          },
        };
      });

      graph.edge(function (edge) {
        return {
          style: {
            // stroke: edge.newadd,
            lineWidth: edge.newadd
          }
        }
      });
      graph.data(data);
      graph.render();

      if (typeof window !== 'undefined')
        window.onresize = () => {
          if (!graph || graph.get('destroyed')) return;
          if (!container || !container.scrollWidth || !container.scrollHeight) return;
          graph.changeSize(container.scrollWidth, container.scrollHeight);
        };
      // const container = document.getElementById(id);
      // const width = container.scrollWidth;
      // const height = container.scrollHeight || 500;
      // const graph = new G6.Graph({
      //   container: id,
      //   width,
      //   height,
      //   layout: {
      //     type: 'force',
      //     preventOverlap: true,
      //     center: [width / 2, height / 2]
      //   },
      //   modes: {
      //     default: ['zoom-canvas', 'drag-canvas', 'drag-node'],
      //   },
      //   defaultNode: {
      //     size: 10,
      //   },
      // });
      // graph.node(function (node) {
      //   // console.log(node.c, node.label, node.id)

      //   return {
      //     label: node.label,
      //     labelCfg: {
      //       style: {
      //         opacity: 0,
      //       },
      //     },
      //     // size: 12,
      //     style: {
      //       fill: getColor1(node.c),
      //       stroke: node.bolder,
      //       lineWidth: (node.bold * 3) / 10
      //     },
      //   };
      // });

      // graph.data(data);
      // graph.render();

      function logNodePositions() {
        const nodes = graph.getNodes();
        const nodeList = [];

        nodes.forEach((node) => {
          const model = node.getModel();
          const nodeData = {
            id: model.id,
            position: [model.x, model.y]
          };
          nodeList.push(nodeData);
        });

        // Convert the list to a JSON string
        const jsonString = JSON.stringify(nodeList, null, 2);
        console.log(jsonString)

        // Save the JSON string to a file
        saveJsonToFile(jsonString, 'nodes.json');
      }
      function saveJsonToFile(jsonString, filename) {
        // Create a blob from the JSON string
        const blob = new Blob([jsonString], { type: 'application/json' });

        // Create a URL for the blob
        const url = URL.createObjectURL(blob);

        // Create a link element
        const link = document.createElement('a');

        // Set the download attribute with the filename
        link.download = filename;

        // Set the href attribute to the blob URL
        link.href = url;

        // Programmatically click the link to trigger the download
        link.click();

        // Clean up by revoking the object URL
        URL.revokeObjectURL(url);
      }

      // Call this function whenever you need to log the positions
      // setTimeout(() => {
      //   logNodePositions();
      // }, 8000)



      // function refreshDragedNodePosition(e) {
      //   const model = e.item.get('model');
      //   model.fx = e.x;
      //   model.fy = e.y;
      //   // logNodePositions();

      // }
      // graph.on('node:dragstart', (e) => {
      //   graph.layout();
      //   refreshDragedNodePosition(e);
      //   // logNodePositions();
      // });
      // graph.on('node:drag', (e) => {
      //   refreshDragedNodePosition(e);
      //   // logNodePositions();
      // });
      // if (typeof window !== 'undefined')
      //   window.onresize = () => {
      //     if (!graph || graph.get('destroyed')) return;
      //     if (!container || !container.scrollWidth || !container.scrollHeight) return;
      //     graph.changeSize(container.scrollWidth, container.scrollHeight);
      //     // logNodePositions();
      //   };

    },

    showg1() {
      const path = 'http://127.0.0.1:5000/getGraph3';
      const msg1 = axios.get(path)
        .then((res) => {
          this.graph1 = res.data.g1

          const data1 = {
            nodes: this.graph1.nodes,
            edges: this.graph1.edges

          };
          const data2 = {
            nodes: res.data.g2.nodes,
            edges: res.data.g2.edges

          };
          const data3 = {
            nodes: res.data.g3.nodes,
            edges: res.data.g3.edges

          };
          const data4 = {
            nodes: res.data.g4.nodes,
            edges: res.data.g4.edges

          };
          const data5 = {
            nodes: res.data.g5.nodes,
            edges: res.data.g5.edges

          };
          console.log(data1)
          this.showgraph(data2, 'container')
          this.showgraph(data3, 'container2')
          // this.showsmallgraph(data3, 'container3')
          // this.showsmallgraph(data4, 'container4')

        })
        .catch((error) => {
          console.error(error);
        });
      return msg1

    },
    showg3() {
      const path = 'http://127.0.0.1:5000/getGraph4';
      const msg1 = axios.get(path)
        .then((res) => {
          this.graph1 = res.data.g1

          const data1 = {
            nodes: this.graph1.nodes,
            edges: this.graph1.edges

          };
          const data2 = {
            nodes: res.data.g2.nodes,
            edges: res.data.g2.edges

          };
          const data3 = {
            nodes: res.data.g3.nodes,
            edges: res.data.g3.edges

          };
          const data4 = {
            nodes: res.data.g4.nodes,
            edges: res.data.g4.edges

          };
          const data5 = {
            nodes: res.data.g5.nodes,
            edges: res.data.g5.edges

          };
          console.log(data1)
          this.showsmallgraph(data1, 'container3')
          this.showsmallgraph(data2, 'container4')
          this.showsmallgraph(data3, 'container5')
          this.showsmallgraph(data4, 'container6')
          // this.showsmallgraph(data4, 'container4')

        })
        .catch((error) => {
          console.error(error);
        });
      return msg1

    },
    // 为了得到坐标
    showg2() {
      const path = 'http://127.0.0.1:5000/getGraph2';
      const msg1 = axios.get(path)
        .then((res) => {
          this.graph1 = res.data


          this.showgraph2(res.data, 'container3')
          // this.showgraph(data2, 'container2')
          // this.showsmallgraph(data3, 'container3')
          // this.showsmallgraph(data4, 'container4')






        })
        .catch((error) => {
          console.error(error);
        });
      return msg1

    },


    showhuihe() {
      const path = 'http://127.0.0.1:5000/getheatdata';
      const msg1 = axios.get(path)
        .then((res) => {
          console.log(res.data)
          // this.graph1 = res.data.g3
          var myChart = echarts.init(document.getElementById('relizhexian'));


          // 折线图的数据
          var lineData = res.data.loss;

          var x_con = res.data.x
          // 热力图数据
          var heatmapDataList = res.data.heatd
          var cl = res.data.l_c



          var option = {

            tooltip: {
              trigger: 'axis',
              position: function (point, params, dom, rect, size) {
                // `point` 是提示框的当前位置
                // `size` 是提示框的尺寸信息
                // `size.contentSize` 是提示框的宽高
                // `size.viewSize` 是图表容器的宽高
                let x = point[0];
                let y = point[1];

                // 向下偏移 20 像素
                y += 30;
                x += 20

                // 保证提示框不超出视图范围
                if (y + size.contentSize[1] > size.viewSize[1]) {
                  y = size.viewSize[1] - size.contentSize[1];
                }
                if (x + size.contentSize[0] > size.viewSize[0]) {
                  x = size.viewSize[0] - size.contentSize[0];
                }
                return [x, y];
              }
            },

            xAxis: [
              {
                type: 'category',
                data: x_con
              }
            ],
            yAxis: [
              {
                type: 'value'
              }
            ],
            series: [
              {
                data: lineData,
                type: 'line',
                lineStyle: {
                  color: '#bbb'  // 设置折线颜色为灰色
                },
                itemStyle: {
                  color: function (params) {
                    var colorList = cl;
                    return colorList[params.dataIndex];
                  }
                }
              }
            ],
            grid: [
              {
                left: '1%',
                right: '0.358%',
                bottom: '10%'
              }
            ],
            visualMap: {
              min: 0,
              max: 5,
              calculable: true,
              orient: 'horizontal',
              show: false,
              // left: 'center',
              // top: '5%'
            }
          };

          // 动态添加热力图
          heatmapDataList.forEach((heatmapData, index) => {
            heatmapData.days.forEach((day, dayIndex) => {
              option.series.push({
                name: day,
                type: 'heatmap',
                data: heatmapData.data.filter(item => item[0] === dayIndex).map(function (item) {
                  return [item[1], item[0], item[2] || '-'];
                }),
                xAxisIndex: index + 1,
                yAxisIndex: index + 1,
                itemStyle: {
                  // padding: 10
                  borderColor: 'white', // 设置边框颜色
                  borderWidth: 0.1 // 设置边框宽度
                },
                emphasis: {
                  itemStyle: {
                    shadowBlur: 10,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                  }
                }
              });
            });

            // 定义热力图的网格
            option.grid.push({
              show: false,
              left: 80 + index * 20,
              top: '29%',
              width: 20,
              height: 127
            });

            // 定义热力图的坐标轴
            option.xAxis.push({
              gridIndex: index + 1,
              type: 'category',
              show: false,
              data: heatmapData.hours,
              splitArea: {
                show: true
              }
            });

            option.yAxis.push({
              gridIndex: index + 1,
              type: 'category',
              show: false,
              data: heatmapData.days,
              splitArea: {
                show: true
              }
            });
          });

          // 更新 visualMap 的 seriesIndex
          option.visualMap.seriesIndex = option.series.map((_, idx) => idx).filter(idx => idx > 0);

          myChart.setOption(option);








        })
        .catch((error) => {
          console.error(error);
        });
      return msg1


    },



    showTimeSeris1(x, data1, name) {
      var chartDom = document.getElementById('tm1');
      var myChart = echarts.init(chartDom);
      var option;
      let legend = name

      let data01 = data1

      let x_o1 = x
      option = {
        legend: {
          data: legend,
          textStyle: {
            fontSize: 10 // 设置字体大小更小
          },
          top: '17%'

        },
        grid: {
          left: '18%',
          right: '5%'
          // 留出空间给x轴标题

          // textStyle: {
          //   fontSize: 10 // 设置字体大小更小
          // },
        },

        xAxis: {
          type: 'category',
          data: x_o1,
          // name: 'Hour of the day', // 添加x轴标题
          // nameLocation: 'middle', // 标题位置
          // nameTextStyle: {
          //   fontSize: 13,
          //   padding: 20 // 标题与轴线的距离
          // }
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: '{value}'
          },
          // name: 'Power (kW)', // 在y轴左上角添加单位
          // nameLocation: 'end', // 标题位置
          // nameTextStyle: {
          //   fontSize: 12, // 设置字体大小
          //   padding: [0, 0, 0, -40] // 标题与轴线的距离，调整padding来定位
          // },
          // axisLabel: {
          //   formatter: '{value}'
          // }
          // name: 'Power', // 添加y轴标题
        },
        series: data01
      };

      option && myChart.setOption(option);

    },
    showTimeSeris2(x, data01, name) {
      var chartDom = document.getElementById('tm3');
      var myChart = echarts.init(chartDom);
      var option;

      let legen = name
      let data102 = data01

      let x_102 = x

      option = {
        legend: {
          data: legen,
          textStyle: {
            fontSize: 10 // 设置字体大小更小
          },
          top: '13%'
        },
        grid: {
          top: '33%',
          // 留出空间给x轴标题
          containLabel: true // 确保图表在容器内
        },
        xAxis: {
          type: 'category',
          data: x_102,
          // name: 'Hour of the day', // 添加x轴标题
          // nameLocation: 'middle', // 标题位置
          // nameTextStyle: {
          //   fontSize: 13,
          //   padding: 20 // 标题与轴线的距离
          // }
        },
        yAxis: {
          type: 'value',
          min: 10
        },
        series: data102
      };

      option && myChart.setOption(option);

    },
    showTimeSeris3(x, data01, name) {
      var chartDom = document.getElementById('tm2');
      var myChart = echarts.init(chartDom);
      var option;

      let legen = name
      let data102 = data01
      let x_102 = x

      option = {
        legend: {
          data: legen,
          textStyle: {
            fontSize: 10 // 设置字体大小更小
          },
          top: '8%'
        },
        grid: {
          top: '30%',
          // 留出空间给x轴标题
          containLabel: true // 确保图表在容器内
        },
        xAxis: {
          type: 'category',
          data: x_102,
          // name: 'Hour of the day', // 添加x轴标题
          // nameLocation: 'middle', // 标题位置
          // nameTextStyle: {
          //   fontSize: 13,
          //   padding: 20 // 标题与轴线的距离
          // }
        },
        yAxis: {
          type: 'value',
          min: -0.8,
          max: 0.5,
        },
        series: data102
      };

      option && myChart.setOption(option);

    },

  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.metrics:hover {
  cursor: pointer;
  background-color: #f7f7f7;
}

.list-group-item {
  padding: 10px
}

.list-group-item {
  cursor: pointer;
}

.red-circle {
  width: 10px;
  height: 10px;
  border: 3px solid #BF444C;
  /* background-color: red; */
  border-radius: 50%;
  display: inline-block;
}

.grey-circle {
  width: 10px;
  height: 10px;
  border: 3px solid #A6A6A6;
  /* background-color: red; */
  border-radius: 50%;
  display: inline-block;
}

.container {
  padding: 20px;
  max-width: 1500px;
}

.nav-link {
  padding: 0.33rem 0.7rem;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 16px;

}

.row {
  display: flex;
  justify-content: space-between;
  width: 1500px;
  /* 调整为space-between确保间隙 */
  /* margin-bottom: 20px; */
}

.box1 {
  border: 1px solid #ccc;
  /* border-bottom: none; */
  border-right: none;
  /* 边框颜色改为浅灰色 */
  width: 26%;
  /* 调整宽度以确保间隙 */
  height: 830px;
  /* 根据需要调整高度 */
  position: relative;
  left: 2px;
  box-sizing: border-box;
  padding: 0;
  /* 包含padding和border在内 */
  /* padding: 10px; */
  /* 给每个方框内增加一些内边距 */
}

.box2 {
  border: 1px solid #ccc;
  /* border-left: none; */
  border-bottom: none;
  /* 边框颜色改为浅灰色 */
  width: 74%;
  /* 调整宽度以确保间隙 */
  height: 240px;
  /* 根据需要调整高度 */
  position: relative;
  box-sizing: border-box;
  left: 2px;
  /* 包含padding和border在内 */
  /* padding: 10px; */
  /* 给每个方框内增加一些内边距 */
}

.box3 {
  position: absolute;
  bottom: 590px;
  left: 392px;

  border: 1px solid #ccc;
  border-bottom: none;

  /* 边框颜色改为浅灰色 */
  width: 74%;
  /* 调整宽度以确保间隙 */
  height: 330px;
  /* 根据需要调整高度 */
  position: relative;
  box-sizing: border-box;
  /* 包含padding和border在内 */
  /* padding: 10px; */
  /* 给每个方框内增加一些内边距 */
}

.box4 {
  position: absolute;
  bottom: 590px;
  left: 392px;
  border: 1px solid #ccc;
  /* 边框颜色改为浅灰色 */
  width: 74%;
  /* 调整宽度以确保间隙 */
  height: 260px;
  /* 根据需要调整高度 */
  position: relative;
  box-sizing: border-box;
  /* 包含padding和border在内 */
  /* padding: 10px; */
  /* 给每个方框内增加一些内边距 */
}

.title {
  background-color: #f0f0f0;
  /* 改为浅灰色背景 */
  color: #333;
  /* 字体颜色改为深灰色 */
  text-align: center;
  width: 100%;
  /* padding: 2px; */
  /* position: absolute;
  top: 0; */
  width: calc(100% - 0px);
  /* 计算宽度以适应内边距 */
  box-sizing: border-box;
  /* 包含padding和border在内 */
  font-family: Arial, Helvetica, sans-serif;
  font-size: large;
  font-weight: bold;
}

.subtitle {
  color: #666;
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 10px;
}


.slider label {
  margin-right: 10px;


}

.slider input[type="range"] {
  vertical-align: middle;
}


input[type="range"] {
  width: 40%;
  -webkit-appearance: none;
  height: 8px;
  border-radius: 4px;
  background: -webkit-linear-gradient(#d6d6d6, #d6d6d6) no-repeat rgb(238, 237, 237);
  background-size: 50% 100%;
  /* 因为周期默认value=0.50正好占50% */
}

/* -webkit-slider-thumb仅对谷歌浏览器有效 */
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  background-color: #d6d6d6;
  width: 8px;
  height: 15px;
  border-radius: 4px;
  cursor: pointer;
}

#control input[type="range"]::-webkit-slider-thumb:hover {
  background: #b6b6b6;
}

/* 左侧渐变色的写法,默认滑块在最左侧所以下面white为0% */

.accordion-panel {
  border: 1px solid #ccc;
  /* Adds a consistent border */
  margin: 0;
  /* Removes any default margin */
  padding: 0;
  /* Removes any default padding */
}

.accordion-header {
  padding: 0.75rem 1.25rem;
  /* Adjusts padding for headers */
  background-color: #f7f7f7;
  /* Light background for headers */
  border-bottom: 1px solid #ccc;
  /* Border to separate header from content */
  cursor: pointer;
  /* Pointer cursor on hover */
  width: 100%;
  /* Ensures the header is full width */
}

.accordion-body {
  padding: 1rem;
  /* Padding inside the body for content */
  width: 100%;
  /* Ensures the body is full width */
}

/* No additional padding or margin for nested accordions to align them fully */
.nested-accordion .accordion-header,
.nested-accordion .accordion-body {
  margin-right: 0;
  padding-right: 0;
}


/* .subtitle2 {
  color: #666;
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 3px;
} */

.custom-file-label::after {
  font-family: Arial, Helvetica, sans-serif;
  color: #75787c;
  font-size: 15px;
  height: 33px;

}

.content {

  padding: 10px;
}

.parameter {

  margin-bottom: 10px;
}

.palabel {
  display: inline;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 15px;
  font-weight: 600;
  color: #75787c;
  text-align: center;
}

/*滚动条高宽度*/
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

/*滚动条滑块*/
::-webkit-scrollbar-thumb {
  border-radius: 3px;
  box-shadow: inset 0 0 5px rgb(247, 246, 246);
  background: #f0efef;
}

/*滚动条里面轨道*/
::-webkit-scrollbar-track {
  box-shadow: 1px 1px 5px rgb(231, 230, 230) inset;

}

::-webkit-scrollbar-thumb:hover {
  background: #d6d6d6;
}

/*滚动条的小边角*/
::-webkit-scrollbar-corner {
  background: transparent;
}

/* .control-panel {
  margin-top: 20px;
} */

/* .parameter-group {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}



.parameter label {
  font-size: 14px;
  color: #555;
  margin-bottom: 5px;
}

.parameter input {
  padding: 5px;
  border: 1px solid #ccc;
  border-radius: 3px;
  width: 50%;
}

.time-series-chart {
  margin-top: 20px;
}

.overview p {
  margin: 5px 0;
  font-size: 14px;
  color: #555;
}

.chart-placeholder {
  border: 1px dashed #ccc;
  padding: 20px;
  text-align: center;
  margin-top: 10px;
  color: #999;
} */

/* 
.parameter {
  margin-bottom: 10px;
}

.time-series-chart {
  margin-top: 20px;
}

.chart-placeholder {
  border: 1px dashed #ccc;
  padding: 20px;
  text-align: center;
  margin-top: 10px;
} */

/* h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
} */
</style>
