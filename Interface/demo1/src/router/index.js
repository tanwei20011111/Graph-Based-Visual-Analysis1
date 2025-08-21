import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'
import Draw from '@/components/Draw'
import Home2 from '../components/Home2.vue'
import Home3 from '../components/Home3.vue'
import Home4 from '../components/Home4.vue'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/system',
      name: 'HelloWorld',
      component: HelloWorld
    },
    {
      path: '/system2',
      name: 'Home',
      component: Home2
    },
    {
      path: '/system3',
      name: 'Home',
      component: Home3
    },

    {
      path: '/draw',
      name: 'Draw',
      component: Draw
    },
    {
      path: '/system4',
      name: 'Home4',
      component: Home4
    }
  ]
})
