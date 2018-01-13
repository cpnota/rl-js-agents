const { createServer } = require('http')

const server = createServer((req, res) => {
  req.pipe(res)
})

if (!module.parent) {
  server.listen(process.env.PORT || 8080)
}
