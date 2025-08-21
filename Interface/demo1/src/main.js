// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
// import '../static/css/style.css';
import '../static/css/bootstrap-4.1.3/css/bootstrap.css'
import '../static/css/bootstrap-4.1.3/js/bootstrap.js';
import '../static/css/bootstrap-4.1.3/js/bootstrap.bundle.js';

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
