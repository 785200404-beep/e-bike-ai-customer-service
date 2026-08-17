// 后端地址：
//   电脑上跑 `python server.py` 后，开发者工具里用 127.0.0.1 即可。
//   真机预览 / 分享给同事时，改成电脑的局域网 IP（手机和电脑要连同一个 Wi-Fi）。
//   查 IP：`ipconfig getifaddr en0`（Mac）
const API = 'http://192.168.1.6:8000'
const SID_KEY = 'cs_session_id'

// 会话 id：生成一次存在本地，之后每次都带同一个 → 客服能"记得"这一场对话
function getSessionId() {
  let sid = wx.getStorageSync(SID_KEY)
  if (!sid) {
    sid = Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8)
    wx.setStorageSync(SID_KEY, sid)
  }
  return sid
}

// 当前时间 HH:MM，显示在每条消息下面
function nowTime() {
  const d = new Date()
  const p = n => (n < 10 ? '0' + n : '' + n)
  return p(d.getHours()) + ':' + p(d.getMinutes())
}

// 从问题里抓"首驱某车型"，留资表单自动预填"想看的车型"
function detectModel(q) {
  const m = q.match(/首驱\s*([A-Za-z0-9][A-Za-z0-9\- ]*)/)
  if (m && m[1].trim()) return '首驱' + m[1].trim()
  return ''
}

// 客户是不是在问"你们店在哪 / 就近门店 / 附近有没有店"（和后端 _LOC_RE 保持一致）。
// 用正则收紧：裸"导航"不拦（"这车有导航吗"是车机导航）；"最近/附近"后必须跟"店/门店/家"
const LOC_RE = /在哪里|在哪|在哪儿|什么位置|位置在|门店地址|店址|门店位置|店的地址|你们地址|地址是|地址在|地址发我|就近|最近.{0,4}(店|门店|家)|离我|附近.{0,4}(店|门店)|怎么走|怎么去|怎么过来|怎么过去|在哪条|导航到|导航过去|导航去|导航过来|导航一下/
function isLocationQuestion(q) {
  return LOC_RE.test(q)
}

// 取客户定位（授权成功后回调坐标；拒绝/失败回调 null，不影响正常对话）
function getUserLocation(cb) {
  wx.getLocation({
    type: 'gcj02',
    success: (res) => cb && cb({ lat: res.latitude, lng: res.longitude }),
    fail: () => cb && cb(null),
  })
}

// 球面距离（公里）——"到店"列表按离客户远近排序
function haversineKm(lat1, lng1, lat2, lng2) {
  const R = 6371
  const rad = x => x * Math.PI / 180
  const dLat = rad(lat2 - lat1), dLng = rad(lng2 - lng1)
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dLng / 2) ** 2
  return R * 2 * Math.asin(Math.sqrt(a))
}

Page({
  data: {
    messages: [
      {
        role: 'bot',
        welcome: true,
        content: '📌 首驱电动车 · 南宁一网\n营业时间：每天 9:00-21:00，周末不休\n以旧换新：旧车抵 300-800 元 ｜ 上牌不收费，只收牌照邮递费 15 元\n可问：价格、续航、库存、算账、售后、跑外卖选车、竞品对比、门店位置',
        time: nowTime()
      }
    ],
    draft: '',
    loading: false,
    // 快捷按钮：对比 / 到店 / 预约看车（带 action 的不发问，直接弹层）
    suggestions: [
      { q: '雅迪和首驱哪个好？', label: '⚡ 和雅迪比' },
      { q: '九号和首驱怎么选？', label: '⚡ 和九号比' },
      { q: '首驱电动车有什么卖点？', label: '💡 首驱卖点' },
      { q: '跑外卖选什么车？', label: '🛵 跑外卖选车' },
      { q: '首驱S300和九号E300P比哪个强？', label: '🏆 旗舰对比' },
      { action: 'store', label: '🏪 到店' },
      { action: 'lead', label: '📱 预约看车' },
    ],
    // 门店列表（南宁一网 13 家）+ 留资
    showStore: false,
    storeList: [],
    showLead: false,
    leadPhone: '',
    leadModel: '',
    leadSaving: false,
  },

  onLoad() {
    // 预热门店列表（地址/电话/营业时间），点"到店"直接用
    this.fetchStoreList()
  },

  noop() {},

  onInput(e) {
    this.setData({ draft: e.detail.value })
  },

  onSuggest(e) {
    const { q, action } = e.currentTarget.dataset
    if (action === 'store') { this.showStore(); return }
    if (action === 'lead') { this.showLead(); return }
    const qq = (q || '').trim()
    if (!qq || this.data.loading) return
    this.setData({ draft: qq })
    this.onSend()
  },

  onSend() {
    const q = (this.data.draft || '').trim()
    if (!q || this.data.loading) return

    const messages = this.data.messages.concat([{ role: 'user', content: q, time: nowTime() }])
    this.setData({ messages, draft: '', loading: true })
    this.lastModel = detectModel(q) || this.lastModel || ''

    // 问"就近/在哪"时先取定位，让客服能算"最近的门店"；取不到就照常发
    if (isLocationQuestion(q)) {
      getUserLocation((loc) => this.doRequest(q, loc))
    } else {
      this.doRequest(q, null)
    }
  },

  doRequest(q, loc) {
    const data = { question: q, session_id: getSessionId() }
    if (loc) { data.lat = loc.lat; data.lng = loc.lng }
    wx.request({
      url: API + '/api/chat',
      method: 'POST',
      header: { 'Content-Type': 'application/json' },
      data: data,
      success: (res) => {
        const d = res.data || {}
        let answer = d.answer || ''
        const tools = d.used_tools || []
        if (tools.length) {
          // used_tools 里既有真工具（计算器结果/库存查询结果），也有确定性护栏标记
          // （SHUNT_REFUSE/COMPETITOR_AVAIL/DELIVERY_ANSWER/AFTER_SALES_ANSWER/LOCATION_ANSWER）。
          // 一一对应徽章，别把护栏也标成"查库存"；认不出的标记跳过。
          const MAP = [
            ['计算器', '🧮 算账'],
            ['库存查询', '📦 查库存'],
            ['SHUNT_REFUSE', '🚫 守则拦截'],
            ['COMPETITOR_AVAIL', '🏁 竞品提醒'],
            ['DELIVERY_ANSWER', '🛵 外卖选车'],
            ['AFTER_SALES_ANSWER', '🔧 售后'],
            ['LOCATION_ANSWER', '📍 就近门店'],
          ]
          const names = []
          tools.forEach(t => {
            for (const [k, label] of MAP) {
              if (t.indexOf(k) >= 0) { names.push(label); return }
            }
          })
          if (names.length) answer += '\n\n（已用工具：' + names.join('、') + '）'
        }
        if (!answer) answer = '（服务返回异常，请检查后端日志）'
        // 位置问法带 mapLink → 聊天气泡下渲染「导航到店」按钮
        this.setData({
          messages: this.data.messages.concat([{
            role: 'bot', content: answer, time: nowTime(), mapLink: d.map_link || null
          }])
        })
      },
      fail: () => {
        this.setData({
          messages: this.data.messages.concat([{
            role: 'bot',
            content: '连不上后端。请先跑：python server.py（开发者工具还要勾选「不校验合法域名」）',
            time: nowTime()
          }])
        })
      },
      complete: () => this.setData({ loading: false }),
    })
  },

  // —— 到店（南宁一网 13 家，按离客户远近排）+ 留资（P0：邀约到店 + 抓潜在客户）——
  fetchStoreList() {
    wx.request({
      url: API + '/api/store',
      success: (res) => {
        const list = (res.data && res.data.stores) || []
        if (list.length) this.setData({ storeList: list })
      },
    })
  },

  showStore() {
    if (!this.data.storeList.length) {
      wx.showToast({ title: '门店信息还没填，老板去 data/stores.json 填', icon: 'none' })
      return
    }
    // 有定位就按距离排，让"最近的门店"排最前
    getUserLocation((loc) => {
      let list = this.data.storeList
      if (loc) {
        list = list.map(s => {
          const d = haversineKm(loc.lat, loc.lng, s.lat, s.lng)
          return Object.assign({}, s, { dist: d })
        }).sort((a, b) => (a.dist - b.dist))
      }
      this.setData({ storeList: list, showStore: true })
    })
  },
  hideStore() { this.setData({ showStore: false }) },
  // 聊天气泡下的「导航到店」按钮：直接打开微信内置地图导航
  onMapTap(e) {
    const { lat, lng, name, address } = e.currentTarget.dataset
    if (!lat || !lng) {
      wx.showToast({ title: '门店位置还没填', icon: 'none' })
      return
    }
    wx.openLocation({
      latitude: Number(lat), longitude: Number(lng),
      name: name || '首驱门店', address: address || '',
    })
  },
  callStore(e) {
    const phone = e.currentTarget.dataset.phone
    if (!phone || phone.indexOf('0000000') >= 0) {
      wx.showToast({ title: '门店电话还没填', icon: 'none' })
      return
    }
    wx.makePhoneCall({ phoneNumber: phone.replace(/-/g, '') })
  },
  navigateStore(e) {
    const { lat, lng, name, address } = e.currentTarget.dataset
    if (!lat || !lng) {
      wx.showToast({ title: '门店位置还没填，老板去 data/stores.json 填', icon: 'none' })
      return
    }
    wx.openLocation({
      latitude: Number(lat), longitude: Number(lng),
      name: name || '首驱门店', address: address || '',
    })
  },

  showLead() {
    this.setData({ leadModel: this.lastModel || '', showLead: true })
  },
  hideLead() { this.setData({ showLead: false }) },
  onLeadPhone(e) { this.setData({ leadPhone: e.detail.value }) },
  onLeadModel(e) { this.setData({ leadModel: e.detail.value }) },
  submitLead() {
    const phone = (this.data.leadPhone || '').trim()
    if (!/^1\d{10}$/.test(phone)) {
      wx.showToast({ title: '手机号要 11 位、1 开头', icon: 'none' })
      return
    }
    this.setData({ leadSaving: true })
    wx.request({
      url: API + '/api/lead',
      method: 'POST',
      header: { 'Content-Type': 'application/json' },
      data: { phone, model: (this.data.leadModel || '').trim() },
      success: (res) => {
        if (res.data && res.data.ok) {
          this.setData({ showLead: false, leadPhone: '', leadSaving: false })
          wx.showModal({
            title: '已登记 ✅',
            content: '门店会尽快联系您，也可以直接到店看车！',
            showCancel: false,
            confirmText: '好的',
          })
        } else {
          wx.showToast({ title: (res.data && res.data.error) || '提交失败', icon: 'none' })
          this.setData({ leadSaving: false })
        }
      },
      fail: () => {
        wx.showToast({ title: '连不上后端', icon: 'none' })
        this.setData({ leadSaving: false })
      },
    })
  },
})
